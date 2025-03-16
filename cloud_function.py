import functions_framework
from cloud_function import JinjaTemplateGenerator


@functions_framework.http
def BusinessCentral_Sync(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    prompt = request_json["user_prompt"]
    variables = request_json["variables"]
    sample_json = request_json["sample_json"]
    existing_template = request_json.get("existing_template")
    detail_level = request_json.get("detail_level")
    return JinjaTemplateGenerator().generate_template(prompt, variables, sample_json, existing_template, detail_level)
