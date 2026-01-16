# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import requests
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReadTool(BaseTool):
    """Read tool for retrieving full law content using external retrieval services.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the read tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ReadTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized ReadTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the read tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing law_name and section_num

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        law_name = parameters.get("law_name")
        title = parameters.get("title")

        if not law_name and not title:
            error_msg = "Error: 'law_name' or 'title' is missing in parameters."
            logger.error(f"[ReadTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        if law_name:
            payload = {"law_name": law_name, "section_num": parameters.get("section_num")}
        else:
            payload = {"title": title}
        
        try:
            response = requests.post(
                self.retrieval_service_url,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                # Try to surface a helpful error message to the model.
                detail = None
                try:
                    body = response.json()
                    detail = body.get("detail") if isinstance(body, dict) else None
                except Exception:
                    detail = None

                msg = detail or f"Read failed with HTTP {response.status_code}"
                return ToolResponse(text=json.dumps({"result": msg})), 0.0, {"http_status": response.status_code}

            result = response.json()
            result_text = result.get("text", "No content found.")
            
            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            return ToolResponse(text=result_text), 0.0, {}

        except Exception as e:
            error_result = json.dumps({"result": f"Read execution failed: {e}"})
            logger.error(f"[ReadTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
