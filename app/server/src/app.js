import express from 'express';
import { api, errorHandler } from './routes/api.js';
import { createWellbeingRouter } from './wellbeing/routes.js';

// Express app without a listener, so tests can drive it directly.
export function createApp({ wellbeing } = {}) {
  const app = express();
  app.disable('x-powered-by');
  app.use('/api/wellbeing', wellbeing ?? createWellbeingRouter());
  app.use('/api', api);
  app.use('/api', (_req, res) => res.status(404).json({ error: 'not_found', message: 'No such API route.' }));
  app.use('/api', errorHandler);
  return app;
}
