import os
import re
import shutil
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
# SAME GEMINI MODEL AS YOUR ORIGINAL CODE
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite-preview"
)


def get_gemini_api_key():
    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key

    # Works in Google Colab if the key is stored in Colab Secrets
    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError(
        "GEMINI_API_KEY not found. Add GEMINI_API_KEY to Render Environment "
        "Variables or Google Colab Secrets."
    )


llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=get_gemini_api_key(),
    temperature=0.1,
    timeout=90,
    max_retries=1,
)


# ============================================================
# STATE
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
# HELPERS
# ============================================================

def message_text(content) -> str:
    """Convert Gemini response content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))

        return "\n".join(parts)

    return str(content)


def clean_code(text: str) -> str:
    """Remove markdown fences if Gemini adds them."""
    text = message_text(text).strip()

    text = re.sub(
        r"^```(?:verilog|systemverilog|sv)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def require_iverilog():
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        raise RuntimeError(
            "Icarus Verilog is not installed. "
            "The Render Docker deployment must contain iverilog."
        )


# ============================================================
# TOOL: COMPILE + SIMULATE VERILOG
# ============================================================

@tool
def run_verilog_simulation(verilog_code: str, testbench_code: str) -> str:
    """
    Compile and simulate Verilog using Icarus Verilog.
    """

    require_iverilog()

    workdir = tempfile.mkdtemp(prefix="agentic_verilog_")

    rtl_file = os.path.join(workdir, "design.v")
    tb_file = os.path.join(workdir, "tb_design.v")
    output_file = os.path.join(workdir, "simulation.out")

    try:
        with open(rtl_file, "w", encoding="utf-8") as f:
            f.write(verilog_code)

        with open(tb_file, "w", encoding="utf-8") as f:
            f.write(testbench_code)

        compile_result = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                output_file,
                rtl_file,
                tb_file,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if compile_result.returncode != 0:
            return (
                "COMPILE_FAILED\n\n"
                f"STDOUT:\n{compile_result.stdout}\n\n"
                f"STDERR:\n{compile_result.stderr}"
            )

        simulation_result = subprocess.run(
            ["vvp", output_file],
            capture_output=True,
            text=True,
            timeout=30,
        )

        return (
            "COMPILE_SUCCESS\n\n"
            f"SIMULATION_RETURN_CODE: {simulation_result.returncode}\n\n"
            f"STDOUT:\n{simulation_result.stdout}\n\n"
            f"STDERR:\n{simulation_result.stderr}"
        )

    except subprocess.TimeoutExpired:
        return "SIMULATION_TIMEOUT: Icarus Verilog exceeded the 30 second limit."

    except Exception as exc:
        return f"SIMULATION_ERROR: {type(exc).__name__}: {exc}"


# ============================================================
# NODE 1: TASK INPUT
# ============================================================

def task_input_node(state: VerilogState):

    print("[1/5] Received hardware requirement.")

    return {
        "messages": [
            HumanMessage(content=state["requirement"])
        ],
        "next_step": "developer",
    }


# ============================================================
# NODE 2: VERILOG RTL DEVELOPER
# ============================================================

def real_time_developer(state: VerilogState):

    print("[2/5] Calling Gemini to generate Verilog RTL...")

    prompt = f"""
You are a senior RTL Verilog designer.

Convert the following natural-language hardware requirement into
synthesizable Verilog RTL.

HARDWARE REQUIREMENT:
{state["requirement"]}

STRICT RULES:
1. Output ONLY Verilog/SystemVerilog source code.
2. Do NOT use markdown code fences.
3. The top-level DUT module MUST be named exactly: design
4. Use clear input and output names.
5. Make the design synthesizable.
6. Include reset/clock/enable only when required by the specification.
7. Do not generate a testbench.
8. Do not provide explanations.
"""

    try:
        response = llm.invoke(prompt)
        code = clean_code(response.content)

        if not code:
            raise RuntimeError("Gemini returned empty RTL code.")

        print("[2/5] RTL generation completed.")

        return {
            "code": code,
            "next_step": "tester",
        }

    except Exception as exc:
        print(f"[2/5] RTL generation failed: {exc}")

        return {
            "code": "",
            "report": (
                "RTL GENERATION FAILED\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
            "next_step": "end",
        }


# ============================================================
# NODE 3: TESTBENCH + SIMULATION + REPORT
# ============================================================

def real_time_tester(state: VerilogState):

    if not state["code"]:
        return {
            "next_step": "end"
        }

    print("[3/5] Calling Gemini to generate Verilog testbench...")

    test_prompt = f"""
You are a Verilog verification engineer.

Generate a self-contained Verilog/SystemVerilog testbench for this RTL.

RTL:
{state["code"]}

STRICT RULES:
1. The DUT module is named exactly: design
2. Instantiate it as: design dut (...)
3. The testbench top module MUST be named: tb_design
4. Include meaningful functional test cases.
5. Include clock/reset stimulus when required.
6. Check expected behavior wherever practical.
7. Print clear PASS/FAIL messages using $display.
8. End simulation with $finish.
9. Output ONLY testbench source code.
10. Do not use markdown code fences.
"""

    try:
        response = llm.invoke(test_prompt)
        testbench = clean_code(response.content)

        if not testbench:
            raise RuntimeError("Gemini returned empty testbench.")

        print("[3/5] Testbench generation completed.")

    except Exception as exc:
        print(f"[3/5] Testbench generation failed: {exc}")

        return {
            "testbench": "",
            "simulation_result": "",
            "report": (
                "TESTBENCH GENERATION FAILED\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
            "next_step": "manager",
        }

    print("[4/5] Compiling and simulating with Icarus Verilog...")

    try:
        simulation_result = run_verilog_simulation.invoke(
            {
                "verilog_code": state["code"],
                "testbench_code": testbench,
            }
        )

        print("[4/5] Verilog simulation completed.")

    except Exception as exc:
        simulation_result = (
            "SIMULATION_ERROR\n\n"
            f"{type(exc).__name__}: {exc}"
        )

        print(f"[4/5] Simulation failed: {exc}")

    print("[5/5] Generating verification report...")

    report_prompt = f"""
You are a Verilog verification engineer.

Analyze the RTL, testbench and actual simulator output below.

RTL:
{state["code"]}

TESTBENCH:
{testbench}

ACTUAL SIMULATOR OUTPUT:
{simulation_result}

Create a concise verification report with:
- Compilation status
- Simulation status
- Tests performed
- PASS/FAIL result
- Errors/warnings
- Suggested fix if something failed

IMPORTANT:
Do not invent test results.
Only report results supported by the actual simulator output.
"""

    try:
        report_response = llm.invoke(report_prompt)
        report = message_text(report_response.content).strip()

    except Exception as exc:
        report = (
            "VERIFICATION REPORT GENERATION FAILED\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Raw simulator output:\n"
            f"{simulation_result}"
        )

    print("[5/5] Verification report completed.")

    return {
        "testbench": testbench,
        "simulation_result": simulation_result,
        "report": report,
        "next_step": "manager",
    }


# ============================================================
# NODE 4: MANAGER
# ============================================================

def manager_node(state: VerilogState):

    print("[Manager] Workflow completed. Sending to archiver.")

    return {
        "next_step": "archiver"
    }


# ============================================================
# NODE 5: ARCHIVER
# ============================================================

def archiver_node(state: VerilogState):

    print("[Archiver] Saving generated files...")

    artifact_dir = os.path.join(
        tempfile.gettempdir(),
        "agentic_verilog_" + uuid.uuid4().hex,
    )

    os.makedirs(artifact_dir, exist_ok=True)

    files = {
        "design.v": state["code"],
        "tb_design.v": state["testbench"],
        "simulation.txt": state["simulation_result"],
        "verification_report.txt": state["report"],
    }

    for filename, content in files.items():
        with open(
            os.path.join(artifact_dir, filename),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

    print(f"[Archiver] Files saved to {artifact_dir}")

    return {
        "artifact_dir": artifact_dir,
        "next_step": "end",
    }


# ============================================================
# SAME LANGGRAPH FLOW
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
# LANGSERVE API SCHEMAS
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


# ============================================================
# LANGSERVE ENTRY POINT
# ============================================================

def run_agentic_verilog(request: DesignRequest) -> DesignResponse:

    print("=" * 60)
    print("AGENTIC VERILOG WORKFLOW STARTED")
    print("=" * 60)
    print(f"Requirement: {request.requirement}")

    try:
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

        print("=" * 60)
        print("AGENTIC VERILOG WORKFLOW FINISHED")
        print("=" * 60)

        return DesignResponse(
            requirement=request.requirement,
            rtl_code=result.get("code", ""),
            testbench=result.get("testbench", ""),
            simulation_result=result.get("simulation_result", ""),
            verification_report=result.get("report", ""),
            artifact_dir=result.get("artifact_dir", ""),
        )

    except Exception as exc:

        print("=" * 60)
        print("WORKFLOW ERROR")
        print("=" * 60)
        print(f"{type(exc).__name__}: {exc}")

        raise RuntimeError(
            f"Agentic Verilog workflow failed: {type(exc).__name__}: {exc}"
        )


verilog_runnable = RunnableLambda(
    run_agentic_verilog
).with_types(
    input_type=DesignRequest,
    output_type=DesignResponse,
)


# ============================================================
# FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="Agentic Verilog RTL Designer",
    version="1.0.0",
    description=(
        "Agentic Verilog RTL generation, testbench generation, "
        "Icarus Verilog simulation and verification."
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
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "iverilog_installed": shutil.which("iverilog") is not None,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
