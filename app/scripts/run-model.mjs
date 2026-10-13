// Starts the Python model service (FastAPI + uvicorn) with the project venv.
// Usage: node scripts/run-model.mjs            -> serve on port 8001
//        node scripts/run-model.mjs --test     -> run the pytest suite instead
// Override the interpreter with PYTHON=C:\path\to\python.exe.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(here, '..');
const modelDir = path.join(appDir, 'model_service');
const venvPython = path.resolve(appDir, '..', 'project', '.venv', 'Scripts', 'python.exe');
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : 'python');
const port = process.env.MODEL_PORT || '8001';

const args = process.argv.includes('--test')
  ? ['-m', 'pytest', '-q', 'tests']
  : ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', port];

console.log(`[model] ${python} ${args.join(' ')}  (cwd ${modelDir})`);
const child = spawn(python, args, { cwd: modelDir, stdio: 'inherit' });
child.on('exit', (code) => process.exit(code ?? 0));
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => child.kill(sig));
