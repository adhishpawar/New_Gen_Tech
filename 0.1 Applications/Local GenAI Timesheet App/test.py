import json
import random

categories = [
    "Bug Fix", "Development", "Documentation", "Testing", "Deployment",
    "Performance", "DevOps", "Refactoring", "Research", "Design", "Analysis", "Security", "Meeting", "Training", "Migration"
]
tasks = [
    "Fixed authentication bug", "Implemented payment gateway", "Wrote API documentation",
    "Tested checkout flow", "Deployed release to production", "Optimized DB indexes",
    "Configured CI/CD pipeline", "Refactored legacy code", "Researched new tech stack",
    "Designed dashboard UI", "Analyzed user feedback", "Patched security issue",
    "Attended sprint planning", "Trained new developer", "Migrated database"
]

with open("synthetic_timesheet_data.jsonl", "w") as f:
    for i in range(10000):
        task = random.choice(tasks) + f" #{random.randint(1,200)}"
        category = random.choice(categories)
        effort = f"{round(random.uniform(1, 8), 1)} hrs"
        entry = {
            "text": task,
            "label": category,
            "effort": effort
        }
        f.write(json.dumps(entry) + "\n")
