from flask import Flask, request
from datetime import datetime
from train import train, evaluate, refresh_csv
import json

app = Flask(__name__)


def stringify(inp_list):
    return ["".join(i) for i in inp_list]


@app.route("/uploadInputDataset", methods=["POST"])
def upload():
    file = request.files["data"]
    with open("./data/malware.csv", "w") as f:
        f.write(file.read().decode("utf-8"))
    refresh_csv()
    print("Running training script ...")
    train()
    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


@app.route("/validate", methods=["POST"])
def validate():
    original_data, final_data = validate()
    original_string_list = stringify(original_data)
    final_string_list = stringify(final_data)
    final_json_list = []
    for original_string, final_string in zip(original_string_list, final_string_list):
        final_json_list.append(
            {"adversarialString": final_string, "originalString": original_string}
        )

    return json.dumps(final_json_list), 200, {"ContentType": "application/json"}


@app.route("/perturbData", methods=["POST"])
def inference():
    original_data = json.loads(request.data)["data"]
    final_data = evaluate(original_data)
    original_string_list = stringify([data[:215] for data in original_data])
    final_string_list = stringify(final_data)
    final_json_list = []
    for original_string, final_string in zip(original_string_list, final_string_list):
        final_json_list.append(
            {"adversarialString": final_string, "originalString": original_string}
        )

    return json.dumps(final_json_list), 200, {"ContentType": "application/json"}


@app.route("/getPermissions", methods=["GET"])
def permissions():
    with open("./data/permissions.txt") as f:
        permissions = f.read().split(",")
    return (
        json.dumps({"permissions": permissions}),
        200,
        {"ContentType": "application/json"},
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
