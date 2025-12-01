from fastapi import FastAPI,requests

app=FastAPI()

@app.post('/get-data')
def keystrokeData():
    data = request.json  # Keystroke data from frontend
    keystrokes = data.get("keystrokes", [])

    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400
    i=1
    while(os.path.exists(f"./data/sample{i}.csv")):
        i+=1
    with open(f"./data/sample{i}.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keystrokes[0].keys())
        writer.writeheader()
        writer.writerows(keystrokes)

    return jsonify({"status": "success", "message": "Keystrokes saved."})
