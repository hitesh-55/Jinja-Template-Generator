import os
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from typing import Optional, List, Dict, Any, Union

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Initialize FastAPI application
app = FastAPI(title="Jinja Template Generator API")

# =============================================================================
# Request, Response & Error Models
# =============================================================================

class TemplateRequest(BaseModel):
    user_prompt: str = Field(
        ...,
        description="User instructions for template generation."
    )
    variables: List[str] = Field(
        ...,
        description="List of variables to be used in the template."
    )
    existing_template: Optional[str] = Field(
        None,
        description="Base64-encoded existing HTML template."
    )
    sample_json: Dict[str, Any] = Field(
        ...,
        description="Sample JSON data to be used as reference for variables in the template."
    )
    detail_level: str = Field(
        "high",
        description="Level of detail in output: 'low', 'medium', or 'high'."
    )

class TemplateResponse(BaseModel):
    jinja_template: str = Field(
        ...,
        description="The generated or modified Jinja template."
    )

class ErrorResponse(BaseModel):
    detail: str

# =============================================================================
# Agent Definitions (Generic)
# =============================================================================

def get_generic_jinja_html_modifier(detail_level: str = "high") -> Agent:
    """
    Returns an Agent configured for modifying an existing generic Jinja template.
    """
    max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
    verbose = detail_level in ["medium", "high"]

    backstory = (
        "You are an expert in modifying Jinja templates for HTML/PDF document generation. "
        "Your task is to update the provided template so that all placeholders exactly match "
        "the keys present in the provided sample JSON and variables list. Do not include any "
        "markdown formatting or code block delimiters in your output. Your generated template "
        "should be pure HTML/Jinja with absolutely no markdown syntax or comments referring to "
        "code blocks. The template should be designed to render beautifully with modern layouts, "
        "attractive fonts, and vibrant colors as specified by the user prompt."
    )

    goal = (
        "Modify the provided Jinja template to strictly adhere to the provided sample JSON structure "
        "and variables list. Ensure that only the specified keys are used, and that the template "
        "features a visually appealing design. Return pure HTML/Jinja code with no markdown."
    )

    return Agent(
        role="Generic Jinja Template Modifier",
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        memory=True,
        allow_delegation=(detail_level == "high"),
        llm="gpt-4o-mini"
    )


def get_generic_jinja_template_creator(detail_level: str = "high") -> Agent:
    """
    Returns an Agent configured for creating a new generic Jinja template.
    """
    max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
    verbose = detail_level in ["medium", "high"]

    backstory = (
        "You are a master in creating Jinja templates for document generation. "
        "Your expertise is in translating user prompts, provided variables, and sample JSON data "
        "into beautiful and dynamic templates. Create templates that work with the exact structure of "
        "the provided sample JSON, being careful to navigate arrays and nested objects correctly. "
        "Never include markdown formatting or code block delimiters in your output. Return only pure "
        "HTML/Jinja code that's ready to be used without any additional processing. Use modern layouts, "
        "appealing fonts, and vibrant colors as directed by the user."
    )

    goal = (
        "Create a detailed and creative Jinja template that strictly adheres to the provided sample JSON "
        "structure and variables list. Every placeholder must exactly match a key from the sample JSON or "
        "the provided variables list. Do not add extraneous content such as markdown or code block delimiters."
    )

    return Agent(
        role="Generic Jinja Template Creator",
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        memory=True,
        allow_delegation=(detail_level == "high"),
        llm="gpt-4o-mini"
    )


# =============================================================================
# Output Parser Agent (Validation and Cleaning)
# =============================================================================

output_parser = Agent(
    role="Output Validator and Cleaner",
    goal=(
        "Ensure the generated Jinja template uses only the keys provided in `sample_json` and `variables`. "
        "Remove any code block delimiters, markdown formatting, or other non-template content from the output. "
        "Return only clean, valid HTML/Jinja template code."
    ),
    backstory=(
        "A strict validator that prevents incorrect placeholders and enforces adherence to sample data. "
        "You also clean the output to ensure it contains only the raw template with no extraneous content."
    ),
    verbose=True,
    memory=True,
    llm="gpt-4o-mini"
)

# =============================================================================
# Task Creation
# =============================================================================

def create_tasks(detail_level: str, agent: Agent, has_existing_template: bool, sample_json: Dict[str, Any], variables: List[str]) -> List[Task]:
    """
    Creates tasks for generating or modifying a Jinja template.
    
    Ensures the generated template:
    - **Strictly follows** the provided `sample_json` structure.
    - **Uses only** variable names from the `variables` list.
    - **Rejects** any assumptions or auto-generated placeholders.
    """
    tasks = []
    
    # Format the variables and JSON structure explicitly for reference
    variables_str = ", ".join(variables)
    
    # Analyze JSON structure to identify arrays that need iteration
    json_structure_info = analyze_json_structure(sample_json)

    if has_existing_template:
        description = (
            f"Modify the provided Jinja template while ensuring it strictly follows the sample JSON data structure "
            f"and uses only these variables: [{variables_str}]. Do NOT introduce any new placeholders or assumptions. "
            f"\n\nJSON structure: {json_structure_info}\n\n"
            f"Return ONLY the pure HTML/Jinja template with no markdown formatting or code block delimiters. "
            f"Your response should start directly with HTML and contain nothing but the template."
        )
    else:
        description = (
            f"Create a detailed Jinja template that strictly adheres to this sample JSON structure "
            f"and uses only these variables: [{variables_str}]. Each placeholder must exactly match one of these keys - "
            f"DO NOT add any extra variables.\n\nJSON structure: {json_structure_info}\n\n"
            f"Return ONLY the pure HTML/Jinja template with no markdown formatting or code block delimiters. "
            f"Your response should start directly with HTML and contain nothing but the template."
        )

    template_task = Task(
        description=description,
        expected_output="A pure HTML/Jinja template with no markdown or code blocks, using only the provided JSON keys and variables list.",
        agent=agent
    )
    tasks.append(template_task)

    if detail_level in ["medium", "high"]:
        validation_task = Task(
            description=(
                f"Clean and validate the generated Jinja template. Ensure it uses ONLY keys from the provided "
                f"sample JSON and these variables: [{variables_str}]. If any undefined variable appears in the template, "
                f"correct it. Strip away any markdown formatting, code block delimiters (like ```), or explanatory text. "
                f"Return ONLY the pure HTML/Jinja template code."
            ),
            expected_output="A clean, fully validated Jinja template with no extraneous content.",
            agent=output_parser
        )
        tasks.append(validation_task)

    return tasks

# =============================================================================
# JSON Analysis Helper
# =============================================================================

def analyze_json_structure(json_data: Dict[str, Any]) -> str:
    """
    Analyzes JSON structure to identify arrays and nested objects.
    Returns a string with description of the structure to help the template generation.
    """
    structure_info = []
    
    def traverse(data, path=""):
        if isinstance(data, dict):
            for k, v in data.items():
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    traverse(v, new_path)
                else:
                    structure_info.append(f"{new_path}: {type(v).__name__}")
        elif isinstance(data, list) and data:
            # For arrays, analyze the first item
            item_type = "empty array"
            if data:
                if isinstance(data[0], (dict, list)):
                    structure_info.append(f"{path} is an array with {len(data)} items")
                    traverse(data[0], f"{path}[0]")
                else:
                    item_type = type(data[0]).__name__
                    structure_info.append(f"{path} is an array of {item_type} with {len(data)} items")
    
    traverse(json_data)
    return "\n".join(structure_info)

# =============================================================================
# Template Cleaner
# =============================================================================

def clean_template_output(result: Union[str, Dict[str, Any]]) -> str:
    """
    Cleans the template output to remove markdown, code blocks, newlines, and other non-template content.
    """
    if isinstance(result, dict) and "template" in result:
        result = result["template"]
    
    if not isinstance(result, str):
        result = str(result)
    
    # Remove common code block markers and explanatory text
    result = result.replace("```html", "").replace("```jinja", "").replace("```", "")
    
    # Remove all newlines
    result = result.replace("\n", "")
    
    # Check for HTML doctype at the beginning and add if missing
    result = result.strip()
    
    return result

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/generate-template/", response_model=TemplateResponse, responses={500: {"model": ErrorResponse}})
async def generate_template(request: TemplateRequest):
    """
    Generates a Jinja template **STRICTLY** based on user instructions, the provided `sample_json`, 
    and the variables list. **No extra placeholders** should be added.
    """
    try:
        detail_level = request.detail_level.lower()
        if detail_level not in ["low", "medium", "high"]:
            detail_level = "high"

        has_existing_template = bool(request.existing_template)
        agent = (get_generic_jinja_html_modifier(detail_level)
                 if has_existing_template
                 else get_generic_jinja_template_creator(detail_level))

        existing_template = ""
        if has_existing_template:
            try:
                existing_template = base64.b64decode(request.existing_template.encode("utf-8")).decode("utf-8")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 encoding in existing_template: {str(e)}"
                )

        if not request.sample_json:
            raise HTTPException(status_code=400, detail="Sample JSON data is required.")

        tasks = create_tasks(
            detail_level, 
            agent, 
            has_existing_template, 
            request.sample_json, 
            request.variables
        )

        jinja_crew = Crew(
            agents=[agent, output_parser],
            tasks=tasks,
            process=Process.sequential,
            verbose=(detail_level == "high")
        )

        inputs = {
            "user_prompt": request.user_prompt,
            "variables": request.variables,
            "existing_template": existing_template,
            "sample_json": request.sample_json,
            "detail_level": detail_level,
            "max_iterations": 3
        }

        result = jinja_crew.kickoff(inputs=inputs)
        
        # Clean the output to ensure it's just the template
        final_template = clean_template_output(result)

        return TemplateResponse(jinja_template=final_template)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template generation failed: {str(e)}")

@app.post("/test-decode/")
async def test_decode(request: TemplateRequest):
    """
    Decodes a base64-encoded template and returns a preview.
    """
    if not request.existing_template:
        return {"status": "No template provided"}

    try:
        decoded = base64.b64decode(request.existing_template.encode("utf-8")).decode("utf-8")
        return {
            "status": "success",
            "length": len(decoded),
            "preview": decoded[:500] + ("..." if len(decoded) > 500 else "")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run the API, use: uvicorn filename:app --reload