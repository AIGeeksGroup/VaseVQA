"""
Skill registration system - ported from stardojo2025
"""
import inspect
import base64
import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Skill:
    skill_name: str
    skill_function: callable
    skill_embedding: Any
    skill_code: str
    skill_code_base64: str

# Global skill storage
SKILLS = {}

def register_skill(name):
    """
    Skill registration decorator - fully consistent with stardojo2025
    """
    def decorator(skill):
        skill_name = name
        skill_function = skill
        skill_code = inspect.getsource(skill)
        
        # Remove decorator to avoid duplication
        if f"@register_skill(\"{name}\")\n" in skill_code:
            skill_code = skill_code.replace(f"@register_skill(\"{name}\")\n", "")
        
        skill_code_base64 = base64.b64encode(skill_code.encode('utf-8')).decode('utf-8')
        
        skill_ins = Skill(
            skill_name=skill_name,
            skill_function=skill_function,
            skill_embedding=None,  # Embedding not needed here
            skill_code=skill_code,
            skill_code_base64=skill_code_base64
        )
        SKILLS[skill_name] = skill_ins
        
        return skill_ins
    
    return decorator

class SkillRegistry:
    """
    Skill registry - simplified version, focused on prompt generation
    """
    
    def __init__(self):
        self.skills = SKILLS
    
    def get_from_skill_library(self, skill_name: str) -> Dict:
        """
        Get skill information from skill library - fully consistent with stardojo2025 format
        """
        if skill_name not in self.skills:
            return None
            
        skill = self.skills[skill_name]
        skill_function = skill.skill_function
        
        docstring = inspect.getdoc(skill_function)
        
        if docstring:
            params = inspect.signature(skill_function).parameters
            
            if len(params) > 0:
                param_descriptions = {}
                
                for param in params.values():
                    name = param.name
                    # Extract parameter description from docstring
                    param_match = re.search(rf"- {name}: (.+)\.", docstring)
                    if param_match:
                        param_descriptions[name] = param_match.group(1)
                    else:
                        param_descriptions[name] = f"Parameter {name}"
                
                res = {
                    "function_expression": f"{skill_name}({', '.join(params.keys())})",
                    "description": docstring,
                    "parameters": param_descriptions,
                }
            else:
                res = {
                    "function_expression": f"{skill_name}()",
                    "description": docstring,
                    "parameters": {},
                }
        else:
            res = None
        
        return res
    
    def get_all_skills(self) -> List[str]:
        """
        Get all skill names
        """
        return list(self.skills.keys())
    
    def get_skill_information(self, skill_list: List[str]) -> List[Dict]:
        """
        Get skill information list - fully consistent with stardojo2025
        """
        filtered_skill_library = []
        
        for skill_name in skill_list:
            skill_item = self.get_from_skill_library(skill_name)
            if skill_item:
                filtered_skill_library.append(skill_item)
        
        return filtered_skill_library

# Global skill registry instance
skill_registry = SkillRegistry()

def get_skill_library() -> List[Dict]:
    """
    Get complete skill library information
    """
    all_skills = skill_registry.get_all_skills()
    return skill_registry.get_skill_information(all_skills)
