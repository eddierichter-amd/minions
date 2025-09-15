from typing import List, Dict, Any, Optional
import json
import time
import re
import os
from datetime import datetime

from minions.minion import Minion, _extract_json
from minions.usage import Usage
from minions.minions_mcp import SyncMCPClient

# Pygame-specific prompts
PYGAME_DECOMPOSITION_PROMPT = """
You are an expert pygame developer. Analyze this game and create detailed specifications for each section:

Task: {task}

Break this into 4-6 sections with specific requirements for each. For each section, provide:
- What exactly needs to be implemented
- Key gameplay mechanics and rules
- Edge cases to handle
- Technical requirements (screen size, colors, physics, etc.)
- Any special behaviors or interactions

Respond with JSON:
{{
    "game_title": "Descriptive Game Name",
    "game_description": "2-3 sentence overview of gameplay",
    "technical_specs": {{
        "screen_width": 800,
        "screen_height": 600,
        "fps": 60,
        "background_color": "BLACK"
    }},
    "sections": [
        {{
            "name": "Game Constants and Setup",
            "requirements": "Define screen dimensions, colors, speeds, sizes. Include all numeric constants the game needs.",
            "specifics": ["SCREEN_WIDTH = 800", "SCREEN_HEIGHT = 600", "List specific colors needed", "Movement speeds", "Object sizes"],
            "edge_cases": ["Ensure constants work for all screen sizes"]
        }},
        {{
            "name": "Player/Main Character Class", 
            "requirements": "Implement the main controllable character with movement, rendering, and collision detection",
            "specifics": ["Movement controls (arrow keys/WASD)", "Position tracking", "Boundary checking", "Visual representation"],
            "edge_cases": ["Screen boundary collisions", "Invalid movement inputs", "Simultaneous key presses"]
        }},
        {{
            "name": "Game Objects Classes",
            "requirements": "Other game entities (enemies, projectiles, collectibles, obstacles)",
            "specifics": ["Individual class for each object type", "Update and draw methods", "Collision rectangles"],
            "edge_cases": ["Objects going off-screen", "Collision detection accuracy", "Spawning/despawning"]
        }},
        {{
            "name": "Game Logic Functions",
            "requirements": "Core game mechanics, scoring, collision handling, game state management",
            "specifics": ["Collision detection between objects", "Scoring system", "Game over conditions", "Level progression if applicable"],
            "edge_cases": ["Multiple simultaneous collisions", "Score overflow", "Game state transitions"]
        }},
        {{
            "name": "Main Game Loop",
            "requirements": "Event handling, game state updates, rendering, and frame rate control",
            "specifics": ["Event processing", "Game object updates", "Screen clearing and drawing", "Frame rate limiting"],
            "edge_cases": ["Window close handling", "Key repeat events", "Performance on slow systems"]
        }}
    ]
}}
"""

SECTION_IMPLEMENTATION_PROMPT = """
🚨 CRITICAL: You are implementing ONE SPECIFIC SECTION of a pygame game. 

🎯 YOUR ASSIGNED SECTION: {section_name}

GAME OVERVIEW: {task}

SECTION REQUIREMENTS: {section_requirements}

IMPLEMENTATION DETAILS:
{section_specifics}

EDGE CASES: {section_edge_cases}

AVAILABLE COMPONENTS FROM PREVIOUS SECTIONS:
{available_components}

 WHAT NOT TO DO:
- Do NOT write a complete game
- Do NOT write multiple sections at once
- Do NOT include imports unless absolutely necessary
- Do NOT write pygame.init() or main() unless this is "Main Game Loop"
- Do NOT copy code from other sections
- Do NOT redefine constants or classes that already exist

WHAT TO DO FOR YOUR SECTION:

**USE EXISTING COMPONENTS**: Reference constants, call functions, and instantiate classes from previous sections where appropriate.

IF section_name == "Game Constants and Setup":
- Write ONLY constants and color definitions
- Example: SCREEN_WIDTH = 800, BLACK = (0, 0, 0)
- NO classes, NO functions, NO pygame.init()

IF section_name == "Player/Main Character Class":  
- Write ONLY the Player class
- Include __init__, update, draw, and collision methods
- USE existing constants for dimensions, colors, speeds
- NO other classes, NO constants redefinition

IF section_name == "Game Objects Classes":
- Write ONLY Enemy and Bullet classes (or similar game objects)
- Each class needs __init__, update, draw methods
- USE existing constants and REFERENCE Player class for interactions
- NO main loop, NO Player class redefinition

IF section_name == "Game Logic Functions":
- Write ONLY helper functions for collision detection, scoring
- Functions like check_collision(), update_score(), etc.
- USE existing classes and constants in your function parameters
- CALL these functions with available class instances

IF section_name == "Main Game Loop":
- Write ONLY the main() function and game loop
- Include pygame.init(), event handling, game loop
- INSTANTIATE all available classes and CALL all available functions
- This is the ONLY section that should have a complete loop

 YOUR SECTION IS: {section_name}

CRITICAL: After your code, provide a detailed COMPONENT DOCUMENTATION section explaining:
- What each function/class/constant you created does
- When other sections should use them  
- What parameters they expect and what they return
- Usage patterns and calling sequences

Format:
```python
# Your code here
```

COMPONENT DOCUMENTATION:
- ConstantName: What it represents, typical values, when to use it
- ClassName: Purpose, key responsibilities, usage pattern
  - method_name(params): What it does, when to call it, what it returns, side effects
  - Required calling sequence: e.g., "Call move() then update()" 
- function_name(param1, param2): Purpose, when to call, what params mean, return value, example usage

Write ONLY what's needed for this specific section, building on existing components:
"""

INTEGRATION_PROMPT = """
You are integrating pygame code sections into one complete, runnable game.

GAME OVERVIEW: {task}

FULL CONTEXT:
{context}

GAME SPECIFICATIONS:
{game_specs}

COMPONENT INVENTORY (ensure all are used appropriately):
{component_inventory}

CODE SECTIONS TO INTEGRATE:
{all_sections}

CRITICAL: You MUST start your response with ```python and end with ```

INTEGRATION REQUIREMENTS:
1. **MANDATORY**: Start your response with ```python and end with ```
2. Ensure proper import statements at the top
3. Initialize pygame correctly
4. Create a proper main() function and game loop
5. Handle all event processing (quit, key presses)
6. Implement proper game state management
7. Add any missing connections between sections
8. Ensure the game runs without errors
9. Include if __name__ == "__main__": main() at the end
10. Make sure that no buttons are reused for different aspects of the game (ex: restart and a game action should be different keys)

CRITICAL COMPONENT USAGE CHECKS:
- All classes from the inventory are properly instantiated in main() according to their documented purpose
- All functions from the inventory are called where appropriate based on their documentation
- All constants from the inventory are used consistently (no redefinition)
- Game objects are updated and drawn in the correct order
- Collision detection works between all relevant objects
- Game over conditions are properly handled
- Screen boundaries are respected using defined constants
- FPS is properly controlled using defined constants

NOTE: The component inventory includes documentation written by the model that created each component, explaining exactly what each does and when to use it.

REMEMBER: Your response must begin with ```python and end with ``` to properly format the code.

Create a complete, polished pygame game file that can be run immediately:
"""

class PygameMinion(Minion):
    """Specialized minion for pygame development with cost-effective modular approach."""
    
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
        
        # Component tracking for cross-section dependencies
        self.created_functions = []  # (name, params, return_desc, section)
        self.created_classes = []    # (name, key_methods, purpose, section)
        self.created_constants = []  # (name, type, purpose, section)
        
    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,
        is_privacy=False,
        images=None,
        is_follow_up=False,
    ):
        """Create a pygame game quickly: remote decomposition -> local sections -> local integration."""
        
        print(f"\n========== PYGAME MINION (FAST) ==========")
        print(f"Task: {task}")
        
        # Reset component tracking for this call
        self.created_functions.clear()
        self.created_classes.clear() 
        self.created_constants.clear()
        
        start_time = time.time()
        local_usage = Usage()
        remote_usage = Usage()
        
        # Initialize conversation log
        conversation_log = {
            "task": task,
            "context": context,
            "approach": "pygame_detailed",
            "conversation": [],
            "generated_final_answer": "",
            "usage": {"remote": {}, "local": {}},
            "detailed_logs": [],
            "sections_implemented": [],
            "timing": {}
        }
        
        # Phase 1: Detailed decomposition (Remote - thorough but single call)
        print("🎯 Phase 1: Detailed decomposition")
        decomp_start = time.time()
        decomp_prompt = PYGAME_DECOMPOSITION_PROMPT.format(task=task)
        decomp_response, decomp_usage = self.remote_client.chat(
            messages=[{"role": "user", "content": decomp_prompt}],
            response_format={"type": "json_object"}
        )
        remote_usage += decomp_usage
        decomp_time = time.time() - decomp_start
        
        try:
            decomposition = json.loads(decomp_response[0])
        except:
            decomposition = _extract_json(decomp_response[0])
        
        sections = decomposition["sections"]
        tech_specs = decomposition["technical_specs"]
        game_context = f"Game: {decomposition['game_title']}\nDescription: {decomposition['game_description']}"
        
        print(f"📋 Game: {decomposition['game_title']}")
        print(f"📋 Sections: {[s['name'] for s in sections]}")
        
        # Log decomposition
        conversation_log["conversation"].append({
            "phase": "decomposition",
            "user": "remote",
            "prompt": decomp_prompt,
            "output": decomp_response[0],
            "model": "remote",
            "timing": decomp_time
        })
        conversation_log["detailed_logs"].append({
            "phase": "decomposition",
            "model": "remote",
            "prompt": decomp_prompt,
            "response": decomp_response[0],
            "usage": decomp_usage.to_dict(),
            "timing": decomp_time,
            "parsed_result": decomposition
        })
        
        # Phase 2: Implement sections with full context (Local)
        print("🔧 Phase 2: Implement sections with detailed requirements")
        section_codes = []
        impl_start = time.time()
        
        for section in sections:
            section_name = section["name"]
            print(f"  Creating: {section_name}")
            section_start = time.time()
            
            # Get available components from previous sections
            available_components = self._format_available_components()
            
            # Build comprehensive context for local model
            try:
                section_prompt = SECTION_IMPLEMENTATION_PROMPT.format(
                    task=task,
                    section_name=section_name,
                    section_requirements=section["requirements"],
                    section_specifics="\n".join(f"- {spec}" for spec in section["specifics"]),
                    section_edge_cases="\n".join(f"- {case}" for case in section["edge_cases"]),
                    available_components=available_components
                )
                
            except Exception as e:
                print(f"❌ ERROR formatting section prompt: {e}")
                raise
            
            section_response, section_usage, _ = self.local_client.chat([
                {"role": "user", "content": section_prompt}
            ])
            local_usage += section_usage
            section_time = time.time() - section_start
            
            # Track components from this section for future sections (using full response)
            self._extract_and_track_components(section_response[0], section_name)
            
            # Extract just the code for the section codes list
            section_code = self._extract_code_block(section_response[0])
            section_codes.append(f"# {section_name.upper()}\n{section_code}")
            
            # Log section implementation
            conversation_log["sections_implemented"].append({
                "section_name": section_name,
                "prompt": section_prompt,
                "response": section_response[0],
                "code": section_code,
                "usage": section_usage.to_dict(),
                "timing": section_time,
                "components_tracked": {
                    "functions": [f['name'] for f in self.created_functions if f['section'] == section_name],
                    "classes": [c['name'] for c in self.created_classes if c['section'] == section_name],
                    "constants": [c['name'] for c in self.created_constants if c['section'] == section_name]
                }
            })
            conversation_log["conversation"].append({
                "phase": f"section_implementation_{section_name}",
                "user": "local", 
                "prompt": section_prompt,
                "output": section_code,
                "model": "local",
                "timing": section_time
            })
        
        impl_time = time.time() - impl_start
        
        # Phase 3: Integration with full specifications (Local)
        print("🔗 Phase 3: Integration with full context")
        all_sections = "\n\n".join(section_codes)
        
        # Format game specs for integration
        game_specs = f"""
Title: {decomposition['game_title']}
Description: {decomposition['game_description']}
Technical Specs: {tech_specs}
        """.strip()
        
        # Get complete component inventory for integration
        component_inventory = self._format_available_components()
        
        try:
            integration_prompt = INTEGRATION_PROMPT.format(
                task=task, 
                context=context,
                game_specs=game_specs,
                component_inventory=component_inventory,
                all_sections=all_sections
            )
            
        except Exception as e:
            print(f"❌ ERROR formatting integration prompt: {e}")
            raise
        
        integration_start = time.time()
        final_response, final_usage, _ = self.local_client.chat([
            {"role": "user", "content": integration_prompt}
        ])
        local_usage += final_usage
        integration_time = time.time() - integration_start
        
        final_code = final_response[0]
        
        # Log integration
        conversation_log["conversation"].append({
            "phase": "integration",
            "user": "local",
            "prompt": integration_prompt,
            "output": final_code,
            "model": "local", 
            "timing": integration_time
        })
        conversation_log["detailed_logs"].append({
            "phase": "integration",
            "model": "local",
            "prompt": integration_prompt,
            "response": final_response[0],
            "usage": final_usage.to_dict(),
            "timing": integration_time
        })
        
        # Final timing and logging
        total_time = time.time() - start_time
        timing = {
            "decomposition_time": decomp_time,
            "implementation_time": impl_time,
            "integration_time": integration_time, 
            "total_time": total_time
        }
        
        conversation_log["generated_final_answer"] = final_code
        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()
        conversation_log["timing"] = timing
        conversation_log["component_inventory"] = {
            "functions": self.created_functions,
            "classes": self.created_classes,
            "constants": self.created_constants
        }
        
        # Save log file like other minions
        if logging_id:
            log_filename = f"{logging_id}_pygame.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}_pygame.json"
        log_path = os.path.join(self.log_dir, log_filename)
        
        print(f"\n=== SAVING PYGAME LOG TO {log_path} ===")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log to {log_path}: {e}")
        
        print(f"\n🎮 Detailed pygame generation complete!")
        print(f"📊 Remote calls: 1, Local calls: {len(sections) + 1}")
        print(f"⏱️  Total time: {timing['total_time']:.2f}s")
        
        # Create supervisor/worker message structure for compatibility
        supervisor_messages = [
            {"role": "user", "content": decomp_prompt},
            {"role": "assistant", "content": decomp_response[0]}
        ]
        
        worker_messages = []
        for section_data in conversation_log["sections_implemented"]:
            worker_messages.extend([
                {"role": "user", "content": section_data["prompt"]},
                {"role": "assistant", "content": section_data["response"]}
            ])
        
        # Add integration messages
        worker_messages.extend([
            {"role": "user", "content": integration_prompt},
            {"role": "assistant", "content": final_response[0]}
        ])
        
        return {
            "final_answer": final_code,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
            "conversation_log": conversation_log,
            "timing": timing,
        }
    
    def _extract_and_track_components(self, section_response: str, section_name: str):
        """Extract and track functions, classes, and constants using the model's own documentation."""
        
        # Split response into code and documentation parts
        code_part, doc_part = self._split_code_and_docs(section_response)
        
        # Parse the model's documentation
        documented_components = self._parse_component_documentation(doc_part)
        
        # Extract actual code components and match with documentation
        function_matches = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:', code_part)
        for name, params, return_type in function_matches:
            clean_params = re.sub(r'=\s*[^,)]+', '', params).strip()
            
            # Use model's documentation or fallback to inference
            purpose = documented_components.get(name, self._infer_function_purpose(name, section_name))
            
            self.created_functions.append({
                'name': name,
                'params': clean_params,
                'return_type': return_type.strip() if return_type else 'None',
                'purpose': purpose,
                'section': section_name
            })
        
        # Extract class definitions
        class_matches = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:(.*?)(?=\nclass|\ndef\s+\w+(?!\s*\(self)|$)', 
                                  code_part, re.DOTALL)
        for name, class_body in class_matches:
            methods = re.findall(r'def\s+(\w+)\s*\([^)]*\):', class_body)
            key_methods = [m for m in methods if not m.startswith('_') or m in ['__init__']]
            
            # Use model's documentation or fallback to inference
            purpose = documented_components.get(name, self._infer_class_purpose(name, section_name))
            
            self.created_classes.append({
                'name': name,
                'key_methods': key_methods,
                'purpose': purpose,
                'section': section_name
            })
        
        # Extract constants
        const_matches = re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)', code_part, re.MULTILINE)
        for name, value in const_matches:
            if self._is_shareable_constant(name, value):
                # Use model's documentation or fallback to inference
                purpose = documented_components.get(name, self._infer_constant_purpose(name))
                const_type = self._infer_constant_type(value)
                
                self.created_constants.append({
                    'name': name,
                    'type': const_type,
                    'purpose': purpose,
                    'section': section_name
                })
    
    def _split_code_and_docs(self, response: str) -> tuple[str, str]:
        """Split the response into code and documentation parts."""
        # Look for COMPONENT DOCUMENTATION section
        doc_match = re.search(r'COMPONENT DOCUMENTATION:\s*(.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        
        if doc_match:
            doc_part = doc_match.group(1)
            # Remove documentation from code part
            code_part = response[:doc_match.start()].strip()
        else:
            code_part = response
            doc_part = ""
        
        # Extract actual code from markdown blocks if present
        code_part = self._extract_code_block(code_part)
        
        return code_part, doc_part
    
    def _parse_component_documentation(self, doc_text: str) -> dict:
        """Parse the model's detailed component documentation into a dictionary."""
        components = {}
        
        doc_lines = doc_text.split('\n')
        i = 0
        
        while i < len(doc_lines):
            line = doc_lines[i].strip()
            
            # Main component (class/function/constant)
            if line.startswith('-') and ':' in line:
                content = line[1:].strip()  # Remove the '-'
                if ':' in content:
                    name_part, desc_part = content.split(':', 1)
                    name = name_part.strip()
                    description = desc_part.strip()
                    
                    # Clean up function names (remove parentheses and parameters)
                    if '(' in name:
                        name = name.split('(')[0].strip()
                    
                    # Look ahead for sub-items (methods, usage patterns)
                    sub_info = []
                    i += 1
                    while i < len(doc_lines):
                        next_line = doc_lines[i].strip()
                        if next_line.startswith('  -') or next_line.startswith('    -'):
                            # This is a sub-item (method or usage note)
                            sub_info.append(next_line.strip())
                            i += 1
                        elif next_line.startswith('-') and ':' in next_line:
                            # New main component, break
                            i -= 1  # Back up one line
                            break
                        elif next_line and not next_line.startswith(' '):
                            # Non-indented text, probably end of this component
                            i -= 1
                            break
                        else:
                            i += 1
                    
                    # Store component with full information
                    full_desc = description
                    if sub_info:
                        full_desc += " | " + " | ".join(sub_info)
                    
                    components[name] = full_desc
            
            i += 1
        
        return components
    
    def _infer_function_purpose(self, name: str, section: str) -> str:
        """Infer function purpose from name and section context."""
        if any(word in name.lower() for word in ['collision', 'collide']):
            return "collision detection"
        elif any(word in name.lower() for word in ['update', 'move']):
            return "object state update"
        elif any(word in name.lower() for word in ['draw', 'render']):
            return "rendering"
        elif any(word in name.lower() for word in ['check', 'validate']):
            return "validation/checking"
        elif 'score' in name.lower():
            return "scoring system"
        else:
            return f"game logic for {section.lower()}"
    
    def _infer_class_purpose(self, name: str, section: str) -> str:
        """Infer class purpose from name and section context."""
        if 'player' in name.lower():
            return "main character control"
        elif any(word in name.lower() for word in ['enemy', 'bullet', 'projectile']):
            return "game object"
        elif 'game' in name.lower():
            return "game state management"
        else:
            return f"entity from {section.lower()}"
    
    def _is_shareable_constant(self, name: str, value: str) -> bool:
        """Determine if a constant is likely to be used by other sections."""
        # Screen dimensions, colors, speeds, sizes are typically shared
        shareable_patterns = ['SCREEN', 'WIDTH', 'HEIGHT', 'COLOR', 'SPEED', 'SIZE', 'FPS']
        return any(pattern in name for pattern in shareable_patterns)
    
    def _infer_constant_type(self, value: str) -> str:
        """Infer the type of a constant from its value."""
        value = value.strip()
        if value.startswith('(') and value.endswith(')'):
            return "tuple/color"
        elif value.isdigit():
            return "integer"
        elif value.replace('.', '').isdigit():
            return "float"
        else:
            return "value"
    
    def _infer_constant_purpose(self, name: str) -> str:
        """Infer constant purpose from name."""
        if 'SCREEN' in name:
            return "screen dimensions"
        elif 'COLOR' in name:
            return "color definition"
        elif 'SPEED' in name:
            return "movement speed"
        elif 'SIZE' in name:
            return "object size"
        else:
            return "game parameter"
    
    def _format_available_components(self) -> str:
        """Format tracked components for use in prompts, using model's detailed documentation."""
        if not (self.created_functions or self.created_classes or self.created_constants):
            return "No components available yet (this is the first section)."
        
        components = []
        
        if self.created_constants:
            components.append("CONSTANTS (use these values, do NOT redefine):")
            for const in self.created_constants:
                components.append(f"  - {const['name']}: {const['purpose']}")
        
        if self.created_classes:
            components.append("\nCLASSES (instantiate these, follow usage patterns):")
            for cls in self.created_classes:
                # Split the detailed documentation 
                purpose_parts = cls['purpose'].split(' | ')
                main_purpose = purpose_parts[0]
                usage_info = purpose_parts[1:] if len(purpose_parts) > 1 else []
                
                components.append(f"  - {cls['name']}: {main_purpose}")
                
                # Add method information
                if cls['key_methods']:
                    components.append(f"    Methods: {', '.join(cls['key_methods'])}")
                
                # Add usage patterns if available
                for usage in usage_info:
                    if 'call' in usage.lower() or 'usage' in usage.lower() or 'sequence' in usage.lower():
                        components.append(f"    USAGE: {usage}")
        
        if self.created_functions:
            components.append("\nFUNCTIONS (call with correct parameters):")
            for func in self.created_functions:
                param_str = f"({func['params']})" if func['params'] else "()"
                
                # Split detailed documentation
                purpose_parts = func['purpose'].split(' | ')
                main_purpose = purpose_parts[0]
                usage_info = purpose_parts[1:] if len(purpose_parts) > 1 else []
                
                components.append(f"  - {func['name']}{param_str}: {main_purpose}")
                
                # Add detailed usage if available
                for usage in usage_info:
                    components.append(f"    {usage}")
        
        return "\n".join(components)

    def _extract_code_block(self, response: str) -> str:
        """Extract code from markdown code blocks."""
        # Try python code blocks first
        for pattern in [r'```python\s*\n(.*?)\n```', r'```py\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```']:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If looks like code (has imports, classes, etc), return as-is
        if any(indicator in response for indicator in ['import ', 'def ', 'class ', 'pygame.']):
            return response.strip()
        
        return response.strip()
    
