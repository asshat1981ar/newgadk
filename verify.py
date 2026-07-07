from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# Custom mock server handler
class MockOllamaHandler(http.server.BaseHTTPRequestHandler):
    chat_calls = 0

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode('utf-8'))
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        if self.path == "/api/chat":
            messages = req.get("messages", [])
            system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
            
            # Determine which agent is calling based on prompt
            if "Clarifying Questions" in user_prompt or "high-impact clarifying questions" in user_prompt:
                # Clarifying questions request
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "1. Output format?\n2. Count characters?\n3. Include README?"
                    }
                }
            elif "Synthesize a detailed" in user_prompt:
                # Goal synthesis request
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "Build a simple word counter in Python that takes text from stdin and prints the word count. Include a README.md and a word_counter.py."
                    }
                }
            elif "Planner" in system_prompt:
                # PLAN phase
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "1. Create word_counter.py\n2. Create README.md\n3. Verify."
                    }
                }
            elif "Architect" in system_prompt:
                # ARCHITECT phase
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "Design notes: Use standard write_file to create word_counter.py."
                    }
                }
            elif "Builder" in system_prompt:
                # BUILD phase
                # First check if the builder is responding to the tools or starting
                if any(m.get("role") == "tool" for m in messages):
                    # Builder has executed tool, now finish
                    res = {
                        "message": {
                            "role": "assistant",
                            "content": "Successfully implemented the word counter tool."
                        }
                    }
                else:
                    # Builder starting: output tool calls to write files
                    res = {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "write_file",
                                        "arguments": {
                                            "path": "word_counter.py",
                                            "content": "import sys\n\ndef main():\n    data = sys.stdin.read()\n    print(len(data.split()))\n\nif __name__ == '__main__':\n    main()\n"
                                        }
                                    }
                                },
                                {
                                    "function": {
                                        "name": "write_file",
                                        "arguments": {
                                            "path": "README.md",
                                            "content": "# Word Counter\nSimple word counting command line tool."
                                        }
                                    }
                                }
                            ]
                        }
                    }
            elif "Critic" in system_prompt:
                # REVIEW phase
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "APPROVE"
                    }
                }
            elif "Governor" in system_prompt:
                # GOVERN phase
                if any(m.get("role") == "tool" for m in messages):
                    res = {
                        "message": {
                            "role": "assistant",
                            "content": "GOVERN: GO"
                        }
                    }
                else:
                    res = {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "run_quality_gates",
                                        "arguments": {}
                                    }
                                }
                            ]
                        }
                    }
            elif "FinOps" in system_prompt:
                # OPERATE phase
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "FinOps: Run spent 1000 tokens."
                    }
                }
            else:
                res = {
                    "message": {
                        "role": "assistant",
                        "content": "Default fallback response."
                    }
                }
            self.wfile.write(json.dumps(res).encode('utf-8'))
        elif self.path == "/api/embeddings" or self.path == "/api/embed":
            # EMBED request
            res = {"embedding": [0.1] * 26}
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self.wfile.write(b"{}")

def main():
    print("Starting mock Ollama API server...")
    server = http.server.HTTPServer(('127.0.0.1', 0), MockOllamaHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Mock server running on port {port}")

    # Temporary directory for output files
    output_dir = os.path.abspath("./verification_output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Set up environment variables
        env = os.environ.copy()
        env["OLLAMA_MODE"] = "direct-cloud"
        env["OLLAMA_CLOUD_HOST"] = f"http://127.0.0.1:{port}"
        env["OLLAMA_API_KEY"] = "dummy_key"
        env["OLLAMA_SWARM_WORKSPACE"] = output_dir

        print("Running assistant...")
        # Launch assistant process using python -m ollama_swarm.assistant
        proc = subprocess.Popen(
            [sys.executable, "-m", "ollama_swarm.assistant", "--output-dir", output_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        try:
            # Feed inputs to stdin
            # 1. Software idea:
            proc.stdin.write("A simple word counter command line tool\n")
            proc.stdin.flush()
            time.sleep(1)

            # 2. Answer to Question 1:
            proc.stdin.write("Simple print statement\n")
            proc.stdin.flush()
            time.sleep(1)

            # 3. Answer to Question 2:
            proc.stdin.write("No, only words\n")
            proc.stdin.flush()
            time.sleep(1)

            # 4. Answer to Question 3:
            proc.stdin.write("Yes\n")
            proc.stdin.flush()

            stdout, stderr = proc.communicate(timeout=30)
            print("\n--- Assistant Output ---")
            print(stdout)
            print("------------------------")
            if stderr:
                print("\n--- Assistant Errors ---")
                print(stderr)
                print("------------------------")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Error: Assistant timed out.")
            sys.exit(1)
        finally:
            server.shutdown()
            server.server_close()

        # Verification: check that word_counter.py and README.md exist in output_dir
        counter_path = os.path.join(output_dir, "word_counter.py")
        readme_path = os.path.join(output_dir, "README.md")

        print("\nVerifying files...")
        if not os.path.exists(counter_path):
            print(f"Error: {counter_path} was not created.")
            sys.exit(1)
        if not os.path.exists(readme_path):
            print(f"Error: {readme_path} was not created.")
            sys.exit(1)
        print("Files exist successfully!")

        # Check syntax of generated Python file
        print("Checking syntax of word_counter.py...")
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", counter_path],
                check=True,
                capture_output=True
            )
            print("Syntax check passed!")
        except subprocess.CalledProcessError as e:
            print(f"Syntax check failed: {e.stderr.decode('utf-8')}")
            sys.exit(1)
    finally:
        # Clean up output dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        print("Verification script successfully completed!")

if __name__ == "__main__":
    main()
