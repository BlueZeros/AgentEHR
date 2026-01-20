import re
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('agentlite.mcp_tools.mcp_builder')

class MCPToolBuilder:

    def _install_package(self, package_name: str) -> bool:
        """Install a package using uv if it's not available."""
        try:
            # Validate package name - skip if it's not a valid package name
            if not package_name or package_name.strip() == "":
                logger.debug(f"Skipping empty package name")
                return False
                
            # Skip common non-package strings that LLMs might generate
            invalid_packages = [
                "no external modules required",
                "none",
                "n/a",
                "not required",
                "no requirements",
                "built-in",
                "standard library",
                "no dependencies",
                "no external imports needed",
                "uses system vision capabilities",
                "built-in only",
                "no imports needed",
                "standard python libraries only",
                "no external packages",
                "python built-ins only"
            ]
            
            if package_name.lower().strip() in [p.lower() for p in invalid_packages]:
                logger.debug(f"Skipping invalid package name: '{package_name}'")
                return False
                
            # Check if it looks like a valid package name (basic validation)
            if not re.match(r'^[a-zA-Z0-9_-]+$', package_name):
                logger.debug(f"Skipping invalid package name format: '{package_name}'")
                return False
            
            logger.info(f"Attempting to install package: {package_name}")
            
            # Try uv first (faster)
            result = subprocess.run(
                ["uv", "add", package_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} with uv")
                return True
            else:
                logger.warning(f"uv installation failed for {package_name}: {result.stderr}")
                
                # Fallback to pip
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed {package_name} with pip")
                    return True
                else:
                    logger.error(f"pip installation also failed for {package_name}: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Installation timeout for package: {package_name}")
            return False
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return False

    def _import_with_auto_install(self, module_name: str) -> Optional[Any]:
        """Try to import a module, installing it automatically if not found."""
        try:
            # Try importing directly first
            import importlib
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {module_name}")
            return module
        except ImportError:
            logger.info(f"Module {module_name} not found, attempting to install...")

            # Common package name mappings
            package_mappings = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'sklearn': 'scikit-learn',
                'bs4': 'beautifulsoup4',
                'yaml': 'PyYAML',
                'requests_html': 'requests-html',
                'dateutil': 'python-dateutil'
            }
            
            # Determine package name
            package_name = package_mappings.get(module_name, module_name)
            
            # Try to install the package
            if self._install_package(package_name):
                try:
                    # Try importing again after installation  
                    module = importlib.import_module(module_name)
                    logger.info(f"Successfully imported {module_name} after installation")
                    return module
                except ImportError as e:
                    logger.error(f"Failed to import {module_name} even after installation: {e}")
                    return None
            else:
                logger.error(f"Failed to install package for module: {module_name}")
                return None

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a flexible global environment with automatic package installation."""
        # Full Python builtins - no restrictions
        import builtins
        safe_globals = {
            '__builtins__': builtins.__dict__.copy()
        }
        
        # Always include commonly used modules
        common_modules = ['os', 're', 'sys', 'json', 'datetime', 'time', 'math', 'random', 'pandas']
        for module_name in common_modules:
            if module_name not in safe_globals:
                module = self._import_with_auto_install(module_name)
                if module:
                    safe_globals[module_name] = module

        logger.info(f"Created globals environment with {len(safe_globals)} available modules")
        return safe_globals

    def parse_builded_tool(self, tool_name, response):
        tool_scripts = response.choices[0].message.content
        if not tool_scripts or tool_scripts.strip() == "":
            return "Error: Failed to build the tool. The response is empty."
        
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, tool_scripts, re.DOTALL)
        if not match:
            return "Error: Failed to build the tool. No valid Python code found in the response."
        output = match.group(1).strip()

        # import some necessary packages
        import_statements = """from fastmcp import Context\nfrom typing import Optional, List, Dict, Any, Annotated\nfrom pydantic import Field\nfrom agentlite.mcp_tools.tool_utils import find_timestamp_column, normalize_datetime, get_resource, get_resource_df"""
        clean_scripts = import_statements + "\n" + output

        # Create a safe execution environment
        safe_globals = self._create_safe_globals()

        try:
            exec(clean_scripts, safe_globals)
        except SyntaxError as e:
            return f"Error: '{tool_name}' created as fallback due to syntax errors"
        except Exception as e:
            return f"Error: {e}"

        if tool_name in safe_globals:
            function = safe_globals[tool_name]
        
        return function