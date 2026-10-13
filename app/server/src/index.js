import http from 'node:http';
import { config } from './config.js';
import { createApp } from './app.js';
import { connectDb, dbStatus } from './db.js';
import { attachLive } from './ws/live.js';

const server = http.createServer(createApp());
attachLive(server);

server.listen(config.port, () => {
  console.log(`[server] http://localhost:${config.port}  (API /api, live socket /ws/live)`);
  console.log(`[server] model service: ${config.modelServiceUrl}`);
});

// The server starts even if the database is down; history and saving report that clearly.
connectDb().then(() => {
  if (!dbStatus().connected) console.warn('[db] not connected: results are not saved, /api/history returns 503');
});

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => server.close(() => process.exit(0)));
}
