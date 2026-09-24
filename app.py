import os
import re
import subprocess
import tempfile
import uuid
from typing import List, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


def get_gemini_api_key() -> str:
    """Read the key from Render/local environment, or Colab Secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError(
        "GEMINI_API_KEY was not found. Set it as a Render environment variable "
        "or add it to Google Colab Secrets."
    )


llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=get_gemini_api_key(),
    temperature=0.1,
)


# ============================================================
# State
# ============================================================

class VerilogState(TypedDict):
    messages: List[BaseMessage]
    next_step: str
    requirement: str
    code: str
    testbench: str
    simulation_result: str
    report: str
    artifact_dir: str


# ============================================================
# Helpers
# ============================================================

def clean_code(text: str) -> str:
    """Remove markdown code fences if Gemini adds them."""
    text = text.strip()
    text = re.sub(r"^```(?:verilog|systemverilog|sv)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def ensure_iverilog() -> None:
    """Give a clear error instead of a confusing subprocess failure."""
    import shutil

    if not shutil.which("iverilog") or not shutil.which("vvp"):
        raise RuntimeError(
            "Icarus Verilog is not installed. "
            "On Colab run: !apt-get update -qq && apt-get install -y iverilog. "
            "For Render, use the supplied Dockerfile."
        )


# ============================================================
# Tool: Verilog compile + simulation
# ============================================================

@tool
def run_verilog_simulation(verilog_code: str, testbench_code: str) -> str:
    """
    Compile and simulate Verilog RTL with Icarus Verilog.
    Returns compiler output and simulation output.
    """
    ensure_iverilog()

    workdir = tempfile.mkdtemp(prefix="agentic_verilog_")
    rtl_path = os.path.join(workdir, "design.v")
    tb_path = os.path.join(workdir, "tb_design.v")
    sim_path = os.path.join(workdir, "simulation.out")

    with open(rtl_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)

    with open(tb_path, "w", encoding="utf-8") as f:
        f.write(testbench_code)

    compile_process = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            sim_path,
            rtl_path,
            tb_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if compile_process.returncode != 0:
        return (
            "COMPILE_FAILED\n"
            f"STDOUT:\n{compile_process.stdout}\n"
            f"STDERR:\n{compile_process.stderr}"
        )

    run_process = subprocess.run(
        ["vvp", sim_path],
        capture_output=True,
        text=True,
        timeout=30,
    )

    return (
        "COMPILE_SUCCESS\n"
        f"SIMULATION_RETURN_CODE: {run_process.returncode}\n"
        f"STDOUT:\n{run_process.stdout}\n"
        f"STDERR:\n{run_process.stderr}"
    )


# ============================================================
# LangGraph nodes
# ============================================================

def task_input_node(state: VerilogState):
    """
    Same role as the original task_input node.
    The requirement comes from the LangServe request instead of input().
    """
    return {
        "messages": [HumanMessage(content=state["requirement"])],
        "next_step": "developer",
    }


def real_time_developer(state: VerilogState):
    requirement = state["requirement"]

    prompt = f"""
You are a senior RTL Verilog designer.

Convert this natural-language hardware requirement into synthesizable Verilog RTL.

REQUIREMENT:
{requirement}

STRICT RULES:
1. Output ONLY Verilog/SystemVerilog source code. No markdown.
2. Use exactly one top-level module named: design
3. Use clear input/output port names.
4. Keep the design synthesizable unless the requirement explicitly needs otherwise.
5. Do not write a testbench.
6. Do not include explanations.
"""

    response = llm.invoke(prompt)
    code = clean_code(response.content)

    return {
        "code": code,
        "next_step": "tester",
    }


def real_time_tester(state: VerilogState):
    code = state["code"]

    test_prompt = f"""
You are a Verilog verification engineer.

Generate a self-contained Verilog/SystemVerilog testbench for the RTL below.

RTL:
{code}

STRICT RULES:
1. The DUT top module is exactly named: design
2. Instantiate it as: design dut (...)
3. The testbench top module must be named: tb_design
4. Generate meaningful test cases covering normal operation and important edge cases.
5. Include clock/reset stimulus when required.
6. Print clear PASS/FAIL messages using $display.
7. End simulation with $finish.
8. Output ONLY testbench source code. No markdown and no explanation.
"""

    tb_response = llm.invoke(test_prompt)
    testbench = clean_code(tb_response.content)

    try:
        simulation_result = run_verilog_simulation.invoke(
            {
                "verilog_code": code,
                "testbench_code": testbench,
            }
        )
    except Exception as exc:
        simulation_result = f"SIMULATION_ERROR\n{exc}"

    report_prompt = f"""
You are a Verilog verification engineer.

Analyze the following RTL, testbench, and simulator output.

RTL:
{code}

TESTBENCH:
{testbench}

SIMULATOR OUTPUT:
{simulation_result}

Create a concise verification report containing:
- Compilation status
- Simulation status
- Tests observed
- PASS/FAIL result
- Errors or warnings
- Suggested RTL/testbench fix if something failed

Do not invent results that are not present in the simulator output.
"""

    report_response = llm.invoke(report_prompt)

    return {
        "testbench": testbench,
        "simulation_result": simulation_result,
        "report": report_response.content.strip(),
        "next_step": "manager",
    }


def manager_node(state: VerilogState):
    """
    REST equivalent of the original interactive manager.
    One API request completes one workflow.
    """
    return {"next_step": "archiver"}


def archiver_node(state: VerilogState):
    """
    Save RTL, testbench and reports for this request.
    The API also returns their contents.
    """
    artifact_dir = os.path.join(
        tempfile.gettempdir(),
        f"agentic_verilog_{uuid.uuid4().hex}",
    )
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "design.v"), "w", encoding="utf-8") as f:
        f.write(state["code"])

    with open(os.path.join(artifact_dir, "tb_design.v"), "w", encoding="utf-8") as f:
        f.write(state["testbench"])

    with open(os.path.join(artifact_dir, "simulation.txt"), "w", encoding="utf-8") as f:
        f.write(state["simulation_result"])

    with open(
        os.path.join(artifact_dir, "verification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(state["report"])

    return {
        "artifact_dir": artifact_dir,
        "next_step": "end",
    }


# ============================================================
# Same LangGraph flow as the original workshop
# ============================================================

workflow = StateGraph(VerilogState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)
workflow.add_node("manager", manager_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")
workflow.add_edge("task_input", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", "archiver")
workflow.add_edge("archiver", END)

rt_app = workflow.compile()


# ============================================================
# LangServe input/output models
# ============================================================

class DesignRequest(BaseModel):
    requirement: str = Field(
        ...,
        min_length=3,
        description="Natural-language hardware design requirement.",
    )


class DesignResponse(BaseModel):
    requirement: str
    rtl_code: str
    testbench: str
    simulation_result: str
    verification_report: str
    artifact_dir: str


def run_agentic_verilog(request: DesignRequest) -> DesignResponse:
    result = rt_app.invoke(
        {
            "messages": [],
            "next_step": "task_input",
            "requirement": request.requirement,
            "code": "",
            "testbench": "",
            "simulation_result": "",
            "report": "",
            "artifact_dir": "",
        },
        config={"recursion_limit": 50},
    )

    return DesignResponse(
        requirement=request.requirement,
        rtl_code=result["code"],
        testbench=result["testbench"],
        simulation_result=result["simulation_result"],
        verification_report=result["report"],
        artifact_dir=result["artifact_dir"],
    )


verilog_runnable = RunnableLambda(run_agentic_verilog).with_types(
    input_type=DesignRequest,
    output_type=DesignResponse,
)


# ============================================================
# FastAPI + LangServe
# ============================================================

app = FastAPI(
    title="Agentic Verilog RTL Designer",
    version="1.0.0",
    description=(
        "LangServe API for natural-language Verilog RTL generation, "
        "testbench generation, Icarus Verilog simulation and verification."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    verilog_runnable,
    path="/verilog",
)


@app.get("/")
def root():
    return {
        "message": "Agentic Verilog pipeline is running.",
        "endpoint": "/verilog/invoke",
        "playground": "/verilog/playground/",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
