from agent import run_agent

if __name__ == "__main__":
    print("File Found. Loading...")
    
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = run_agent(query)
        print("Agent:", result["output"])
