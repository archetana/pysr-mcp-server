#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { PySRWrapper } from './pysr-wrapper.js';
import { tools } from './tools.js';

class PySRMCPServer {
  private server: Server;
  private pysrWrapper: PySRWrapper;

  constructor() {
    // Get Python path from environment or use default
    const pythonPath = process.env.PYTHON_PATH || process.env.PYSR_PYTHON || 'python';

    this.pysrWrapper = new PySRWrapper(pythonPath);

    this.server = new Server(
      {
        name: 'pysr-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
    this.setupErrorHandling();
  }

  private setupErrorHandling(): void {
    this.server.onerror = (error) => {
      console.error('[MCP Error]', error);
    };

    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupHandlers(): void {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return { tools };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const { name, arguments: args } = request.params;

        if (!args) {
          throw new Error('No arguments provided');
        }

        switch (name) {
          case 'fit_symbolic_regression': {
            const result = await this.pysrWrapper.fit(
              args.X as number[][],
              args.y as number[],
              {
                niterations: args.niterations as number | undefined,
                populations: args.populations as number | undefined,
                population_size: args.population_size as number | undefined,
                binary_operators: args.binary_operators as string[] | undefined,
                unary_operators: args.unary_operators as string[] | undefined,
                maxsize: args.maxsize as number | undefined,
                maxdepth: args.maxdepth as number | undefined,
                timeout_in_seconds: args.timeout_in_seconds as number | undefined,
              }
            );

            if (!result.success) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Error: ${result.error}`,
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: `Model trained successfully!\n\nModel path: ${result.model_path}\n\n${result.message}\n\nUse this model_path with other tools to get equations, make predictions, or export equations.`,
                },
              ],
            };
          }

          case 'get_equations': {
            const result = await this.pysrWrapper.getEquations(
              args.model_path as string
            );

            if (!result.success) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Error: ${result.error}`,
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: `Discovered equations:\n\n${JSON.stringify(result.equations, null, 2)}`,
                },
              ],
            };
          }

          case 'predict': {
            const result = await this.pysrWrapper.predict(
              args.model_path as string,
              args.X as number[][],
              args.equation_index as number | undefined
            );

            if (!result.success) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Error: ${result.error}`,
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: `Predictions:\n\n${result.message}`,
                },
              ],
            };
          }

          case 'export_equation': {
            const result = await this.pysrWrapper.exportEquation(
              args.model_path as string,
              args.format as 'latex' | 'sympy' | 'torch' | 'jax',
              args.equation_index as number | undefined
            );

            if (!result.success) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Error: ${result.error}`,
                  },
                ],
                isError: true,
              };
            }

            return {
              content: [
                {
                  type: 'text',
                  text: `Equation in ${args.format} format:\n\n${result.message}`,
                },
              ],
            };
          }

          default:
            return {
              content: [
                {
                  type: 'text',
                  text: `Unknown tool: ${name}`,
                },
              ],
              isError: true,
            };
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing tool: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async run(): Promise<void> {
    // Initialize PySR wrapper
    await this.pysrWrapper.initialize();

    const transport = new StdioServerTransport();
    await this.server.connect(transport);

    console.error('PySR MCP server running on stdio');
  }
}

const server = new PySRMCPServer();
server.run().catch(console.error);
