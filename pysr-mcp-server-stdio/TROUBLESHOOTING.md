# Troubleshooting: MCP Server Not Showing in Claude Code

## Problem
You added the PySR MCP server with `claude mcp add`, but the tools aren't showing up in your current Claude Code session.

## Solution: Restart Claude Code

**MCP servers are only loaded when Claude Code starts.** You need to restart Claude Code to see the new tools.

### Steps to Fix:

1. **Exit your current Claude Code session**
   - Press `Ctrl+D` or type `exit`
   - Or close the terminal window

2. **Start a new Claude Code session**
   ```bash
   cd C:\UJ\PYSR
   claude
   ```

3. **Verify MCP server is loaded**
   - The server should start automatically when Claude Code launches
   - Check for any error messages during startup

4. **Test the tools are available**
   - Ask Claude: "What MCP tools do you have access to?"
   - Or ask: "Can you list all available tools?"
   - You should see 4 PySR tools listed

## Verification Checklist

✅ **Check 1: MCP Configuration**
Your `.claude.json` should have this entry (it does!):
```json
"mcpServers": {
  "pysr-mcp": {
    "type": "stdio",
    "command": "node",
    "args": [
      "C:\\UJ\\PYSR\\pysr-mcp-server\\dist\\index.js"
    ],
    "env": {}
  }
}
```

✅ **Check 2: Built Files Exist**
```bash
dir C:\UJ\PYSR\pysr-mcp-server\dist\
```
You should see `index.js` and other compiled files.

✅ **Check 3: Python Dependencies Installed**
```bash
python -c "import numpy, pysr, pandas; print('All dependencies installed!')"
```

## Common Issues

### Issue 1: Server Fails to Start

**Symptoms**: Error messages when Claude Code starts

**Solutions**:
- Check Node.js is installed: `node --version`
- Check the dist folder exists and has index.js
- View MCP server status: `claude mcp list`

### Issue 2: Python Errors

**Symptoms**: Tools appear but fail when called

**Solutions**:
- Install Python packages: `pip install numpy pysr pandas`
- Check Python path: `which python` (Linux/Mac) or `where python` (Windows)
- Specify Python path in config:
  ```bash
  claude mcp remove pysr-mcp
  claude mcp add pysr-mcp --env PYTHON_PATH=C:\path\to\python.exe -- node C:\UJ\PYSR\pysr-mcp-server\dist\index.js
  ```

### Issue 3: Tools Not Visible

**Symptoms**: No PySR tools available after restart

**Solutions**:
1. Check MCP server list:
   ```bash
   claude mcp list
   ```

2. Check for errors in MCP server logs:
   ```bash
   claude --mcp-debug
   ```

3. Test the server directly:
   ```bash
   cd C:\UJ\PYSR\pysr-mcp-server
   node test_mcp_client.js
   ```

## Manual Verification

You can verify the MCP server works by running our test client:

```bash
cd C:\UJ\PYSR\pysr-mcp-server
node test_mcp_client.js
```

This will:
1. Connect to the MCP server
2. List available tools (should show 4 tools)
3. Test each tool with sample data
4. Show you the results

## Expected Behavior After Restart

When you restart Claude Code in the `C:\UJ\PYSR` directory, you should see:

1. **Startup**: MCP server starts automatically (may see "PySR MCP server running")
2. **Tools Available**: 4 new tools accessible to Claude:
   - `fit_symbolic_regression`
   - `get_equations`
   - `predict`
   - `export_equation`

3. **Usage**: You can now ask Claude things like:
   - "Train a symbolic regression model on data_complex.csv"
   - "Use the fit_symbolic_regression tool with my data"
   - "Find the best equation for my dataset"

## Quick Test Command

After restarting Claude Code, try this:

```
"Please use the fit_symbolic_regression tool to train a model on the data in data_for_mcp.json"
```

Claude should be able to call the tool directly.

## Still Not Working?

If the tools still don't appear after restarting:

1. **Check the config file manually**:
   ```bash
   type C:\Users\Ujjwal\.claude.json | findstr pysr
   ```

2. **Try removing and re-adding**:
   ```bash
   claude mcp remove pysr-mcp
   claude mcp add pysr-mcp node C:\UJ\PYSR\pysr-mcp-server\dist\index.js
   ```

3. **Check server logs** (if available):
   ```bash
   claude --mcp-debug
   ```

4. **File an issue**: If nothing works, the test client output can help diagnose:
   ```bash
   cd C:\UJ\PYSR\pysr-mcp-server
   node test_mcp_client.js > test_output.txt 2>&1
   ```

## Summary

**The fix is simple**: Just restart Claude Code!

MCP servers are loaded on startup, not dynamically during a session.

```bash
# Exit current session
exit

# Start new session in your project directory
cd C:\UJ\PYSR
claude
```

Then ask Claude to list available tools and you should see the PySR tools!
