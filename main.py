import os
import base64
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from typing import Optional, List

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Initialize FastAPI
app = FastAPI(title="Jinja Template Generator API")

# 📌 Define Request Models
class TemplateRequest(BaseModel):
    user_prompt: str = Field(..., description="User instructions for template generation")
    variables: List[str] = Field(..., description="List of variables to be used in template")
    existing_template: Optional[str] = Field(None, description="Base64-encoded existing HTML template")
    detail_level: str = Field("high", description="Level of detail in output: 'low', 'medium', or 'high'")
    sample_json: Optional[dict] = Field(None, description="Sample JSON to be used in the template")

class ErrorResponse(BaseModel):
    detail: str

# 🚀 JinjaHTMLModifier Agent with detailed backstory
def get_jinja_html_modifier(detail_level: str = "high"):
    max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
    verbose = detail_level in ["medium", "high"]
    
    backstory = (
        "With extensive experience in handling Jinja HTML content, JinjaHTMLModifier "
        "seamlessly parses and modifies given HTML while preserving its original design "
        "and embedded templating logic. Every modification is applied exactly as specified, "
        "Do NOT add any comments or newlines to the output."
        "without introducing unintended changes. Working at a {detail_level} detail level "
        f"with up to {max_iterations} iterations."
    )
    
    return Agent(
        role="Jinja HTML Modifier",
        goal="Parse the provided {existing_template} and apply modifications as specified in {user_prompt} without altering any other aspects of the design. Post-process the output to ensure no newline characters are present.",
        backstory=backstory,
        verbose=verbose,
        memory=True,
        allow_delegation=detail_level == "high",
        llm="gpt-4o-mini"
    )

# 🚀 JinjaHTMLCreator Agent
def get_jinja_html_creator(detail_level: str = "high"):
    max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
    verbose = detail_level in ["medium", "high"]
    
    backstory = (
        "With extensive experience in handling Jinja HTML content, JinjaHTMLCreator "
        "seamlessly creates new HTML templates based on the user prompt and provided variables. "
        "The resulting template is highly detailed and comprehensive. "
        "Consider using {variables} in the template to make it dynamic. "
        "Also make sure to see in {sample_json} how the variables are structured. "
        "Also make sure you are working at a {detail_level} detail level. "
        "Also make sure to see if the arrays are used to get iterations. "
        "Do NOT add any comments or newlines to the output."
        "without introducing unintended changes. Working at a {detail_level} detail level "
        f"with up to {max_iterations} iterations."
    )
    
    return Agent(
        role="Jinja HTML Creator",
        goal="Provide a very high detailed template using {sample_json} {variables} {user_prompt} to ensure it is accurate and creative. Post-process the output to ensure no newline characters are present.",
        backstory=backstory,
        verbose=verbose,
        memory=True,
        allow_delegation=detail_level == "high",
        llm="gpt-4o-mini"
    )

# 🚀 Dummy Data Generator Agent
def get_dummy_data_agent(detail_level: str = "high"):
    verbose = detail_level in ["medium", "high"]
    backstory = (
        "As the go-to expert in generating dummy content, you adeptly interpret JSON schemas "
        "and leverage GPT to produce data that fits the schema's format of {sample_json}. Your "
        "deep understanding of translation between JSON structures and human-readable dummy data "
        "makes you essential in rapid prototyping and testing environments."
        "Make sure you give this json output based on {jinja_template} and {sample_json}."
    )
    
    return Agent(
        role="Dummy Data Generator",
        goal="Receive an input JSON schema {sample_json} and generate a dummy JSON response using GPT capabilities",
        backstory=backstory,
        verbose=verbose,
        memory=True,
        allow_delegation=detail_level == "high",
        llm="gpt-4o-mini"
    )

# 📌 Output Parser Agent
output_parser = Agent(
    role="Output Parser",
    goal="Refine, validate, and enrich Jinja templates for correctness and optimal detail.",
    backstory=(
        "An expert in Jinja template design, tasked with ensuring that the final output "
        "maintains all the detailed aspects, inline documentation, and extended structural clarity. "
        "Ensure that no significant details are lost or oversimplified in the final template."
        "Do NOT add any comments or newlines to the output."
    ),
    verbose=True,
    memory=True,
    llm="gpt-4o-mini"
)

# Create tasks dynamically based on detail level and whether an existing template is provided
def create_tasks(detail_level: str, jinja_agent: Agent, has_existing_template: bool):
    tasks = []
    
    if has_existing_template:
        description = (
            f"Detail level: {detail_level}."
            """Receive {existing_template} and {user_prompt}. Parse the
    original Jinja HTML while preserving its design and templating logic. Identify
    and apply exactly the modifications described in {user_prompt}—which may
    include changing text, adding attributes, modifying styles, or rearranging variables—without
    altering any content not mentioned in the prompt. Finally, remove all newline
    characters from the output to ensure a single-line result. Do NOT add any comments or newlines to the output."""
        )
    else:
        description = (
            f"Detail level: {detail_level}. "
            "Generate a new, highly detailed and comprehensive Jinja template based on the user prompt and the provided variables. "
            "Ensure that the template is fully documented with inline comments and detailed explanations for each section."
        )
    
    template_creation_task = Task(
        description=description,
        expected_output="A fully formatted and extensively detailed Jinja template as a string.",
        agent=jinja_agent
    )
    tasks.append(template_creation_task)
    
    if detail_level in ["medium", "high"]:
        validation_description = (
            f"Detail level: {detail_level}. "
            "Analyze the generated Jinja template to ensure correct syntax, proper structure, and optimal readability. "
            "Critically assess the output to guarantee that all intricate details, inline comments, and documentation are preserved and enhanced. "
            "Do not simplify or shorten any content; instead, enrich the template with further clarification where needed."
        )
        template_validation_task = Task(
            description=validation_description,
            expected_output="A validated, refined, and comprehensively detailed Jinja template.",
            agent=output_parser
        )
        tasks.append(template_validation_task)
    
    return tasks

def clean_template_output(result) -> str:
    """
    Cleans the template output to remove markdown, code blocks, and extraneous newlines.
    """
    if isinstance(result, dict) and "template" in result:
        result = result["template"]
    
    if not isinstance(result, str):
        result = str(result)
    
    result = result.replace("```html", "").replace("```jinja", "").replace("```", "")
    result = result.replace("\n", "")
    
    return result.strip()

def clean_json_output(result) -> dict:
    """
    Cleans the JSON output to remove markdown, code blocks, and extraneous newlines.
    """
    if isinstance(result, dict) and "json" in result:
        result = result["json"]
    
    if not isinstance(result, str):
        result = str(result)
    
    result = result.replace("```json", "").replace("```", "")
    result = result.replace("\n", "")
    
    return result.strip()

# 🚀 API Endpoint to Generate Jinja Template and Dummy JSON Data
@app.post("/generate-template/", response_model=dict, responses={500: {"model": ErrorResponse}})
async def generate_template(request: TemplateRequest):
    try:
        # Normalize and validate detail level
        detail_level = request.detail_level.lower()
        if detail_level not in ["low", "medium", "high"]:
            detail_level = "high"  # Default to high if invalid
        
        # Handle existing template if provided
        existing_template = ""
        has_existing_template = False
        if request.existing_template:
            try:
                existing_template = base64.b64decode(request.existing_template.encode("utf-8")).decode("utf-8")
                has_existing_template = True
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid base64 encoding in existing_template: {str(e)}"
                )
        
        # Choose appropriate Jinja agent:
        # Use modifier if an existing template is provided; otherwise, create a new template.
        jinja_agent = get_jinja_html_modifier(detail_level) if has_existing_template else get_jinja_html_creator(detail_level)
        
        # Create dynamic tasks for the Jinja template generation
        tasks = create_tasks(detail_level, jinja_agent, has_existing_template)
        
        jinja_crew = Crew(
            agents=[jinja_agent, output_parser],
            tasks=tasks,
            process=Process.sequential,
            verbose=detail_level == "high"
        )
        
        # Prepare inputs for both crews
        inputs = {
            "user_prompt": request.user_prompt,
            "variables": request.variables,
            "existing_template": existing_template,
            "detail_level": detail_level,
            "sample_json": request.sample_json
        }
        
        # Generate the Jinja template
        jinja_result = jinja_crew.kickoff(inputs=inputs)
        jinja_result = clean_template_output(jinja_result)
        
        # Generate dummy JSON data if sample_json is provided
        dummy_result = None
        if request.sample_json:
            dummy_agent = get_dummy_data_agent(detail_level)
            dummy_task = Task(
                description=(
                    f"Detail level: {detail_level}. "
                    "Generate dummy JSON data based on the provided sample JSON schema {sample_json}. "
                    "Ensure the output is valid JSON without any commentary or newline characters."
                ),
                expected_output="A valid dummy JSON object.",
                agent=dummy_agent
            )
            inputs["jinja_template"] = jinja_result
            dummy_crew = Crew(
                agents=[dummy_agent],
                tasks=[dummy_task],
                process=Process.sequential,
                verbose=detail_level == "high"
            )
            dummy_result = dummy_crew.kickoff(inputs=inputs)
            # Optionally, try to parse the result as JSON:
            try:
                dummy_result = clean_json_output(dummy_result)
                dummy_result = json.loads(dummy_result)
            except Exception:
                # If parsing fails, return the raw output
                pass
        
        # Return output based on detail level
        response = {
            "jinja_template": jinja_result,
            "detail_level": detail_level,
            "variables_used": request.variables,
            "success": True
        }
        if request.sample_json:
            response["dummy_json"] = dummy_result
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template generation failed: {str(e)}")

# Debugging endpoint to verify template parsing
@app.post("/test-decode/")
async def test_decode(request: TemplateRequest):
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

# Run with: uvicorn filename:app --reload
