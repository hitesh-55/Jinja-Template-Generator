import os
import base64
from crewai import Agent, Task, Crew, Process
from typing import Optional, List, Dict, Any, Union

class JinjaTemplateGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        if not self.api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
        
        self.output_parser = Agent(
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
    
    def get_generic_jinja_html_modifier(self, detail_level: str = "high") -> Agent:
        max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
        verbose = detail_level in ["medium", "high"]

        return Agent(
            role="Generic Jinja Template Modifier",
            goal=(
                "Modify the provided Jinja template to strictly adhere to the provided sample JSON structure "
                "and variables list. Ensure that only the specified keys are used, and that the template "
                "features a visually appealing design. Return pure HTML/Jinja code with no markdown."
            ),
            backstory=(
                "You are an expert in modifying Jinja templates for HTML/PDF document generation. "
                "Your task is to update the provided template so that all placeholders exactly match "
                "the keys present in the provided sample JSON and variables list."
            ),
            verbose=verbose,
            memory=True,
            allow_delegation=(detail_level == "high"),
            llm="gpt-4o-mini"
        )
    
    def get_generic_jinja_template_creator(self, detail_level: str = "high") -> Agent:
        max_iterations = {"low": 1, "medium": 2, "high": 7}.get(detail_level, 7)
        verbose = detail_level in ["medium", "high"]

        return Agent(
            role="Generic Jinja Template Creator",
            goal=(
                "Create a detailed and creative Jinja template that strictly adheres to the provided sample JSON "
                "structure and variables list. Every placeholder must exactly match a key from the sample JSON or "
                "the provided variables list. Do not add extraneous content such as markdown or code block delimiters."
            ),
            backstory=(
                "You are a master in creating Jinja templates for document generation. "
                "Your expertise is in translating user prompts, provided variables, and sample JSON data "
                "into beautiful and dynamic templates."
            ),
            verbose=verbose,
            memory=True,
            allow_delegation=(detail_level == "high"),
            llm="gpt-4o-mini"
        )
    
    def get_agent(self, detail_level: str, modify: bool) -> Agent:
        return self.get_generic_jinja_html_modifier(detail_level) if modify else self.get_generic_jinja_template_creator(detail_level)
    
    def create_tasks(self,detail_level: str, agent: Agent, has_existing_template: bool, sample_json: Dict[str, Any], variables: List[str]) -> List[Task]:
        """
        Creates tasks for generating or modifying a Jinja template.
        
        Ensures the generated template:
        - **Strictly follows** the provided `sample_json` structure.
        - **Uses only** variable names from the `variables` list.
        - **Rejects** any assumptions or auto-generated p laceholders.
        """
        tasks = []
        
        # Format the variables and JSON structure explicitly for reference
        variables_str = ", ".join(variables)
        
        # Analyze JSON structure to identify arrays that need iteration
        json_structure_info = self.analyze_json_structure(sample_json)

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
                agent=self.output_parser
            )
            tasks.append(validation_task)

        return tasks
    
    def analyze_json_structure(self, json_data: Dict[str, Any]) -> str:
        structure_info = []
        
        def traverse(data, path=""):
            if isinstance(data, dict):
                for k, v in data.items():
                    new_path = f"{path}.{k}" if path else k
                    traverse(v, new_path)
            elif isinstance(data, list) and data:
                structure_info.append(f"{path} is an array")
                traverse(data[0], f"{path}[0]")
        
        traverse(json_data)
        return "\n".join(structure_info)
    
    def clean_template_output(self,result: Union[str, Dict[str, Any]]) -> str:
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
    
    def generate_template(self, user_prompt: str, variables: List[str], sample_json: Dict[str, Any], existing_template: Optional[str] = None, detail_level: str = "high") -> str:
        
        
        has_existing_template = bool(existing_template)
        agent = self.get_agent(detail_level.lower(), modify=bool(existing_template))
        existing_template_decoded = ""
        if has_existing_template:
            try:
                existing_template_decoded = base64.b64decode(existing_template.encode("utf-8")).decode("utf-8")
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding in existing_template: {str(e)}")
        
        if not sample_json:
            raise ValueError("Sample JSON data is required.")
        
        tasks = self.create_tasks(detail_level, agent, bool(existing_template), sample_json, variables)
        
        jinja_crew = Crew(
            agents=[agent, self.output_parser],
            tasks=tasks,
            process=Process.sequential,
            verbose=(detail_level == "high")
        )
        
        inputs = {
            "user_prompt": user_prompt,
            "variables": variables,
            "existing_template": existing_template_decoded,
            "sample_json": sample_json,
            "detail_level": detail_level,
            "max_iterations": 3
        }
        result = jinja_crew.kickoff(inputs=inputs)
        response = self.clean_template_output(result)
        return {"jinja_template": response}
    
    def test_decode(self, existing_template: str) -> Dict[str, Any]:
        if not existing_template:
            return {"status": "No template provided"}
        try:
            decoded = base64.b64decode(existing_template.encode("utf-8")).decode("utf-8")
            return {"status": "success", "length": len(decoded), "preview": decoded[:500] + ("..." if len(decoded) > 500 else "")}
        except Exception as e:
            return {"status": "error", "message": str(e)}
