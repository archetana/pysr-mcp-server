import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

class MCPClient:
     def _init_(self):
       self.session - None
       self.exit_stack - None

     async def initialize(self, url):
        self._stream_context= streamablehttp_client(url)
        r_stream, w_stream, _= await self._stream_context.__aenter__()
        self._session_context= ClientSession(r_stream, w_stream)
        self.session= await self._session_context.__aenter__()
        await self.session.initialize()

     async def list_tools(self):
        response =await self.session.list_tools()
        tools= response.tools
        return tools
     
     async def cleanup(self):
         if self._session_context:
            await self._session_context.__aexit__(None,None,None)

         if self._stream_context:
            await self._stream_context.__aexit__(None,None,None)

async def main():
     
     client = MCPClient()
     try:
         await client.initialize("http://localhost:8000/mcp")
     
         tools = await client.list_tools()
         print("Available tools:", tools)
     
         # if "predict" in tools:
         response = await client.session.call_tool("predict", {"values": [3.0]})
         print("Prediction for input [3.0]:", response)
     
         # if "best_equation" in tools:
         response = await client.session.call_tool("best_equation", {})
         print("Best equation found:", response)

     finally: 
           await client.cleanup()
     
asyncio.run(main())

# import asyncio
# from mcp.client import MCPClient

# async def run_client():
#     client = MCPClient(server_command="python", server_args=["pysr_server.py"])
#     await client.start()

#     # List available tools
#     tools = await client.list_tools()
#     print("Available tools:", [t.name for t in tools])

#     # Call prediction tool
#     pred = await client.call_tool("predict", {"values": [2.0]})
#     print("Prediction for x=2.0:", pred)

#     # Get best equation
#     eq = await client.call_tool("best_equation", {})
#     print("Best equation found:", eq)

# asyncio.run(run_client())
