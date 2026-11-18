import net from 'node:net';
import { spawn } from 'node:child_process';

function findFreePort(start) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', () => resolve(findFreePort(start + 1)));
    server.once('listening', () => server.close(() => resolve(start)));
    server.listen(start, '127.0.0.1');
  });
}

const apiPort = await findFreePort(Number(process.env.FACEID_API_PORT || 8000));
const webPort = await findFreePort(Number(process.env.FACEID_WEB_PORT || 5173));
const env = { ...process.env, VITE_API_URL: `http://127.0.0.1:${apiPort}` };

console.log(`[dev] API: http://127.0.0.1:${apiPort}`);
console.log(`[dev] Web: http://127.0.0.1:${webPort}`);

const api = spawn('python3', ['-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--reload', '--host', '127.0.0.1', '--port', String(apiPort)], { env, stdio: 'inherit' });
const web = spawn('npm', ['--prefix', 'frontend', 'run', 'dev', '--', '--host', '127.0.0.1', '--port', String(webPort)], { env, stdio: 'inherit' });

let shuttingDown = false;
function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  api.kill('SIGTERM');
  web.kill('SIGTERM');
  setTimeout(() => process.exit(code), 250);
}
process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
api.on('exit', code => { if (!shuttingDown && code !== 0) shutdown(code || 1); });
web.on('exit', code => { if (!shuttingDown && code !== 0) shutdown(code || 1); });
