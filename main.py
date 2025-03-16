import os
import base64
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

class ErrorResponse(BaseModel):
    detail: str

# 🚀 JinjaHTMLModifier Agent with detailed backstory
def get_jinja_html_modifier(detail_level: str = "high"):
    # Increase iterations for high detail to encourage more refinement
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
        llm="gpt-4o-mini"  # Use gpt-4o-mini model
    )

# 📌 Output Parser Agent with instructions to preserve details
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
    llm="gpt-4o-mini"  # Use gpt-4o-mini model
)

# Create tasks dynamically based on detail level and whether an existing template is provided
def create_tasks(detail_level: str, jinja_html_modifier: Agent, has_existing_template: bool):
    tasks = []
    
    # Adjust description based on the presence of an existing template
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
        agent=jinja_html_modifier
    )
    tasks.append(template_creation_task)
    
    # Add validation task only for medium or high detail levels
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
    
    # Remove common code block markers and explanatory text
    result = result.replace("```html", "").replace("```jinja", "").replace("```", "")
    
    # Remove all newlines to return a single continuous template string
    result = result.replace("\n", "")
    
    return result.strip()
# 🚀 API Endpoint to Generate Jinja Template
@app.post("/generate-template/", response_model=dict, responses={500: {"model": ErrorResponse}})
async def generate_template(request: TemplateRequest):
    try:
        # Normalize and validate detail level
        detail_level = request.detail_level.lower()
        if detail_level not in ["low", "medium", "high"]:
            detail_level = "high"  # Default to high if invalid
        
        jinja_html_modifier = get_jinja_html_modifier(detail_level)
        
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
        
        # Create dynamic tasks based on detail level and existing template flag
        tasks = create_tasks(detail_level, jinja_html_modifier, has_existing_template)
        
        # Create the crew with dynamic configuration
        jinja_crew = Crew(
            agents=[jinja_html_modifier, output_parser],
            tasks=tasks,
            process=Process.sequential,  # Tasks run one after the other
            verbose=detail_level == "high"
        )
        
        # Prepare inputs for the crew
        inputs = {
            "user_prompt": request.user_prompt,
            "variables": request.variables,
            "existing_template": existing_template,
            "detail_level": detail_level
        }
        
        result = jinja_crew.kickoff(inputs=inputs)
        result = clean_template_output(result)
        # Return output based on detail level
        if detail_level == "low":
            return {"jinja_template": result}
        else:
            return {
                "jinja_template": result,
                "detail_level": detail_level,
                "variables_used": request.variables,
                "success": True
            }
    
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