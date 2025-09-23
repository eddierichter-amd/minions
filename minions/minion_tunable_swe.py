"""
Mini-SWE-Agent specialized minion for handling software engineering tasks
with proper termination detection and bash command formatting.
"""

import json
import time
import os
import pprint
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from minions.minion import Minion, _extract_json
from minions.usage import Usage
from minions.minions_mcp import SyncMCPClient
from minions.prompts.minion_swe import (
    MINISWE_WORKER_SYSTEM_PROMPT,
    MINISWE_SUPERVISOR_INITIAL_PROMPT,
    MINISWE_SUPERVISOR_CONVERSATION_PROMPT,
    MINISWE_SUPERVISOR_FINAL_PROMPT,
    MINISWE_TASK_ROUTER_PROMPT,
)


class MiniSweMinion(Minion):
    """
    Specialized minion for mini-swe-agent tasks that handles:
    1. Proper detection of termination requests in context
    2. Ensuring exactly one bash command output
    3. Correct formatting for mini-swe-agent protocol
    4. Two-phase routing: local model extracts THOUGHT, then decides routing
    
    Key difference from regular minion tunable:
    - Uses LOCAL model for routing decisions (not remote)
    - First extracts thinking portion only, then decides routing
    - Optimized for mini-swe-agent's THOUGHT + bash command pattern
    """
    
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=3,
        callback=None,
        log_dir="minion_logs",
        mcp_client: SyncMCPClient | None = None,
        is_multi_turn=False,
        max_history_turns=10,
    ):
        super().__init__(
            local_client=local_client,
            remote_client=remote_client,
            max_rounds=max_rounds,
            callback=callback,
            log_dir=log_dir,
            mcp_client=mcp_client,
            is_multi_turn=is_multi_turn,
            max_history_turns=max_history_turns,
        )
        
        self.local_model_name = local_client.model_name if local_client else "unknown"
        self.remote_model_name = remote_client.model_name if remote_client else "unknown"
        
        # Add iterative coordination state
        self.task_context = {}
        self.performance_metrics = {
            "local_successes": 0,
            "local_attempts": 0,
            "remote_successes": 0,
            "remote_attempts": 0,
            "escalations": 0
        }
    

    def _decide_routing(self, message_history: List[Dict]) -> Tuple[str, Usage, str]:
        """
        Decide routing using two-phase approach optimized for mini-swe-agent:
        
        Phase 1: Extract THOUGHT portion only using local model
        - Prompts local model to analyze what needs to be done
        - Gets thinking/reasoning without wasting resources on bash commands
        
        Phase 2: Route based on extracted thought using local model
        - Uses extracted thought as input to routing decision
        - Uses LOCAL model for routing (key difference from regular minion tunable)
        - Returns "local" or "remote" based on complexity analysis
        
        This prevents waste of generating full bash commands before routing decision.
        
        Returns:
            tuple[str, Usage, str]: (routing_decision, total_usage_for_routing, extracted_thought)
        """
        print("🤔 Extracting THOUGHT portion from task using local model...")
        
        # Initialize usage tracking
        total_routing_usage = Usage()
        
        # First, ask the model to analyze what the next step should be
        thinking_extraction_prompt = """You are following the mini-swe-agent step-by-step methodology. Based on the conversation so far, what should be the VERY NEXT SINGLE STEP?

CRITICAL: This is ONLY for planning - DO NOT write any code or commands. Your response will be automatically processed to extract only the reasoning.

## Recommended Workflow (follow in order):
1. **Analyze the codebase** by finding and reading relevant files
2. **Create a script to reproduce the issue** 
3. **Edit the source code** to resolve the issue
4. **Verify your fix works** by running your script again
5. **Test edge cases** to ensure your fix is robust

Look at where you are in this workflow and identify the immediate next action needed.

STOP: Do NOT include ```bash blocks, commands, or code. Only reasoning.

THOUGHT: [Your analysis of what the exact next step should be in the workflow - be specific about the immediate action needed. If this is the beginning, start with analyzing the codebase. Don't skip steps or jump to writing complete solutions.]

Remember: This is PLANNING ONLY. No commands will be executed from this response."""

        # Get thinking from local model
        enhanced_message_history = message_history.copy()
        enhanced_message_history.append({"role": "user", "content": thinking_extraction_prompt})
        thinking_response, thinking_usage, _ = self.local_client.chat(enhanced_message_history)
        total_routing_usage += thinking_usage  # Track thinking extraction usage
        
        # Extract the THOUGHT portion from the response
        full_response = thinking_response[0]
        
        # Parse out the THOUGHT section 
        thought_match = re.search(r'THOUGHT:\s*(.*)', full_response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            extracted_thought = f"THOUGHT: {thought_match.group(1).strip()}"
        else:
            # Fallback: use the full response if no THOUGHT section found, but prefix with THOUGHT:
            extracted_thought = f"THOUGHT: {full_response.strip()}"
        
        print(f"💭 Extracted THOUGHT: {extracted_thought}...")
        print(f"💰 Thinking extraction usage: {thinking_usage}")
        
        # Now use the extracted thought for routing decision (similar to _decide_model_for_turn)
        context_length = len(str(message_history))
        
        router_prompt = MINISWE_TASK_ROUTER_PROMPT.format(
            task=extracted_thought,  # Use the extracted thinking as the task
            context_length=context_length,
            doc_metadata="Mini-SWE-Agent task routing decision",
            local_model_name=self.local_model_name,
            remote_model_name=self.remote_model_name
        )
        
        # Use LOCAL model for routing decision (key difference from regular minion tunable)
        routing_messages = [{"role": "user", "content": router_prompt}]
        try:
            decision_response, decision_usage, _ = self.local_client.chat(
                routing_messages, 
                response_format={"type": "json_object"}
            )
            total_routing_usage += decision_usage  # Track routing decision usage
            decision_json = json.loads(decision_response[0])
            
            print(f"🔄 EXTRACTED SUBTASK: {extracted_thought[:100]}...")
            print(f"🚗 ROUTING DECISION (using LOCAL model): ")
            pprint.pprint(decision_json)
            print(f"💰 Routing decision usage: {decision_usage}")
            print(f"💰 Total routing usage: {total_routing_usage}")
            
            routing_decision = decision_json["routing_decision"]
            
            # Update performance metrics based on decision
            if routing_decision == "local":
                self.performance_metrics["local_attempts"] += 1
            else:
                self.performance_metrics["remote_attempts"] += 1
                self.performance_metrics["escalations"] += 1
                
            return routing_decision, total_routing_usage, extracted_thought
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing routing decision: {e}")
            print(f"Raw response: {decision_response[0] if 'decision_response' in locals() else 'No response'}")
            
            # Still need to track usage even in fallback case
            if 'decision_usage' in locals():
                total_routing_usage += decision_usage
                print(f"💰 Total routing usage (with error): {total_routing_usage}")
            
            # Fallback: if task contains completion signals, go local; otherwise remote
            print("🔄 Fallback: Complex task, routing to REMOTE") 
            return "remote", total_routing_usage, extracted_thought

    def _run_local_direct(self, message_history: List[Dict]):
        """
        Run simple tasks directly on local model without remote supervision.
        Optimized for efficiency and cost reduction.
        """
        print("🏃 Running LOCAL DIRECT (no remote supervision needed)")
        
        start_time = time.time()
        local_usage = Usage()  

        # Get response from local model
        response, usage, _ = self.local_client.chat(message_history)
        local_usage += usage
        
        final_answer = response[0]
        
        # Update performance metrics
        self.performance_metrics["local_successes"] += 1
        self.performance_metrics["local_attempts"] += 1
        
        print(f"✅ LOCAL DIRECT completed in {time.time() - start_time:.1f}s")
        
        timing = {
            "local_call_time": time.time() - start_time,
            "remote_call_time": 0,
            "total_time": time.time() - start_time,
            "overhead_time": 0
        }
        
        return {
            "final_answer": final_answer,
            "supervisor_messages": [],
            "worker_messages": message_history,
            "remote_usage": Usage(),
            "local_usage": local_usage,
            "timing": timing,
            "conversation_log": {
                "task": message_history[-1]["content"] if message_history and message_history[-1]["role"] == "user" else "No user message found",
                "context": message_history,
                "conversation": [
                    {
                        "user": "local",
                        "prompt": message_history[-1]["content"] if message_history and message_history[-1]["role"] == "user" else "No user message found",
                        "context": message_history,
                        "output": final_answer
                    }
                ],
                "generated_final_answer": final_answer,
                "usage": {"local": local_usage.to_dict(), "remote": {}},
                "detailed_logs": [  # Add detailed_logs field to match regular tunable
                    {
                        "model": "local",
                        "input": message_history,
                        "output": [final_answer],
                        "token_count": local_usage.to_dict()
                    }
                ]
            }
        }

    def _run_remote_direct(self, message_history: List[Dict]):
        """
        Run simple tasks directly on remote model without loops or back-and-forth.
        Works like _run_local_direct but uses remote model - no iterative supervisor-worker protocol.
        Optimized for complex tasks that need remote model capability.
        """
        print("🏃 Running REMOTE DIRECT (no loops, just direct remote execution)")
        
        start_time = time.time()
        local_usage = Usage()
        remote_usage = Usage()
        
        # Prepare messages - handle Anthropic vs OpenAI format differences
        # Anthropic clients need system messages as separate parameter, not in messages array
        if hasattr(self.remote_client, 'client') and 'anthropic' in str(type(self.remote_client.client)).lower():
            # Anthropic format: system as separate parameter
            """if images:
                # For Anthropic with images, we need to structure content differently
                user_content = [{"type": "text", "text": task}]
                if images:
                    for image in images:
                        user_content.append({"type": "image", "source": image})
                messages[0]["content"] = user_content"""
            
            # Get response from remote model with system parameter
            system_prompt = message_history[0]["content"]

            task_histories = message_history[1:]
            remote_start_time = time.time()
            response, usage = self.remote_client.chat(task_histories, system=system_prompt)
        else:
            # OpenAI format: system messages in messages array
            
            # Get response from remote model
            remote_start_time = time.time()
            response, usage = self.remote_client.chat(message_history)
        timing = {
            "local_call_time": 0,
            "remote_call_time": time.time() - remote_start_time,
            "total_time": time.time() - start_time,
            "overhead_time": 0
        }
        remote_usage += usage
        
        final_answer = response[0]
        
        # Update performance metrics
        self.performance_metrics["remote_successes"] += 1
        self.performance_metrics["remote_attempts"] += 1
        
        print(f"✅ REMOTE DIRECT completed in {time.time() - start_time:.1f}s")
        
        # Calculate timing after completion
        timing["total_time"] = time.time() - start_time
        timing["overhead_time"] = timing["total_time"] - timing["remote_call_time"]
        
        # Initialize conversation log (matching format)
        conversation_log = {
            "task": message_history[-1]["content"] if message_history and message_history[-1]["role"] == "user" else "No user message found",
            "context": message_history,
            "conversation": [
                {
                    "user": "remote",
                    "prompt": message_history[-1]["content"] if message_history and message_history[-1]["role"] == "user" else "No user message found",
                    "context": message_history,
                    "output": final_answer
                }
            ],
            "generated_final_answer": final_answer,
            "usage": {"remote": remote_usage.to_dict(), "local": local_usage.to_dict()},
            "detailed_logs": [  # Add detailed_logs field to match format
                {
                    "model": "remote",
                    "input": message_history,
                    "output": [final_answer],
                    "token_count": remote_usage.to_dict()
                }
            ]
        }
        
        return {
            "final_answer": final_answer,
            "supervisor_messages": message_history,  # Store the message_history for compatibility
            "worker_messages": [],
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "conversation_log": conversation_log,
            "timing": timing,
        }

    def __call__(
        self,
        message_history: List[Dict]
    ):
        """
        Main entry point for mini-swe-agent tasks with smart routing.
        Uses local-direct for simple tasks, remote-direct for complex tasks.
        No loops or iterative protocols - just direct execution based on routing decision.
        """
        print(f"\n========== MINI-SWE-AGENT MINION STARTED ==========")
        
        # Smart routing decision
        routing_decision, initial_routing_usage, extracted_thinking = self._decide_routing(message_history)

        # Create enhanced message history with extracted thinking as assistant context
        enhanced_message_history = message_history
        """enhanced_message_history = message_history.copy()
        if extracted_thinking:
            print(f"💭 Adding extracted thinking to message history: {extracted_thinking[:100]}...")
            # Add the extracted thinking as an assistant message to provide context
            enhanced_message_history.append({
                "role": "assistant", 
                "content": f"Let me think about this step by step:\n\n{extracted_thinking}"
            })
        """

        if routing_decision == "local":
            # Only truly trivial commands go direct to local - no supervision needed
            print("🚀 Using LOCAL DIRECT execution (trivial command, no supervision)")
            result = self._run_local_direct(enhanced_message_history)

            # Add initial routing usage to local usage
            result["local_usage"] += initial_routing_usage
            # Update conversation log usage
            result["conversation_log"]["usage"]["local"] = result["local_usage"].to_dict()
            
            # Add initial routing decision to detailed logs
            result["conversation_log"]["detailed_logs"].insert(0, {
                "model": "local",
                "operation": "initial_routing_decision", 
                "output": [f"Initial routing decision: {routing_decision}"],
                "token_count": initial_routing_usage.to_dict()
            })
        else:
            # All other tasks use remote direct execution (no loops)
            print("🤝 Using REMOTE DIRECT execution (complex task, no loops)")
            result = self._run_remote_direct(enhanced_message_history)

            # Add initial routing usage to local usage (routing uses local model)
            result["local_usage"] += initial_routing_usage
            # Update conversation log usage
            result["conversation_log"]["usage"]["local"] = result["local_usage"].to_dict()
            
            # Add initial routing decision to detailed logs
            result["conversation_log"]["detailed_logs"].insert(0, {
                "model": "local",
                "operation": "initial_routing_decision",
                "output": [f"Initial routing decision: {routing_decision}"],
                "token_count": initial_routing_usage.to_dict()
            })

        # Save log (matching regular tunable minion format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract a safe task name from the last user message
        last_user_message = message_history[-1]["content"] if message_history and message_history[-1]["role"] == "user" else "unknown_task"
        
        # Handle different content formats (string vs list of content blocks)
        if isinstance(last_user_message, list):
            # Anthropic-style content blocks - extract text from all text blocks
            content_text = ""
            for block in last_user_message:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_text += block.get("text", "")
            last_user_message = content_text if content_text else "unknown_task"
        elif not isinstance(last_user_message, str):
            # Convert other types to string
            last_user_message = str(last_user_message)
        
        safe_task = re.sub(r"[^a-zA-Z0-9_\s]", "_", last_user_message[:15])  # Take first 15 chars and sanitize
        safe_task = re.sub(r"\s+", "_", safe_task)  # Replace spaces with underscores
        log_filename = f"{timestamp}_{safe_task}.json"
            
        log_path = os.path.join(self.log_dir, log_filename)

        print(f"\n=== SAVING LOG TO {log_path} ===")  # Match regular tunable message
        try:
            with open(log_path, "w", encoding="utf-8") as f:  # Match encoding settings
                json.dump(result["conversation_log"], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log to {log_path}: {e}")  # Match error message format

        # Add log_file to result (matching regular tunable format)
        result["log_file"] = log_path

        return result
