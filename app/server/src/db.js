import mongoose from 'mongoose';
import { config } from './config.js';

// Connects to a local MongoDB, or to an in-memory one for development (see README).
const status = { connected: false, mode: null, error: null };

async function connectLocal() {
  await mongoose.connect(config.mongoUri, { serverSelectionTimeoutMS: 3000 });
  status.mode = 'local';
}

async function connectMemory() {
  // Downloads a MongoDB binary the first time it runs. Data is lost when the server stops.
  const { MongoMemoryServer } = await import('mongodb-memory-server');
  const server = await MongoMemoryServer.create();
  await mongoose.connect(server.getUri());
  status.mode = 'memory';
}

export async function connectDb() {
  const tryMemory = config.mongoMode === 'memory' || config.mongoMode === 'auto';
  if (config.mongoMode !== 'memory') {
    try {
      await connectLocal();
    } catch (err) {
      status.error = `local MongoDB not reachable: ${err.message}`;
      if (!tryMemory || config.mongoMode === 'local') {
        console.warn(`[db] ${status.error}`);
        return status;
      }
      console.warn(`[db] ${status.error}; falling back to mongodb-memory-server`);
    }
  }
  if (!status.mode && tryMemory) {
    try {
      await connectMemory();
    } catch (err) {
      status.error = `in-memory MongoDB failed: ${err.message}`;
      console.warn(`[db] ${status.error}`);
      return status;
    }
  }
  status.connected = mongoose.connection.readyState === 1;
  status.error = status.connected ? null : status.error;
  if (status.connected) console.log(`[db] connected (${status.mode})`);
  return status;
}

export function dbStatus() {
  return { ...status, connected: mongoose.connection.readyState === 1 };
}

export function dbReady() {
  return mongoose.connection.readyState === 1;
}
