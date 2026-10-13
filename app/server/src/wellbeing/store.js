// Session store. Sessions live in memory while active and are persisted to MongoDB whenever the existing
// database connection is up (same setup as the rest of the app). Without a database the module still works.
import { WellbeingSession } from './WellbeingSession.js';
import { dbReady } from '../db.js';

export function createStore({ model = WellbeingSession, isDbReady = dbReady } = {}) {
  const cache = new Map();
  const owns = (s, key) => s.userId === key || s.anonymousId === key;
  return {
    async get(id) {
      if (cache.has(id)) return cache.get(id);
      if (!isDbReady()) return null;
      const doc = await model.findOne({ sessionId: id }).lean();
      if (!doc) return null;
      cache.set(id, doc);
      return doc;
    },
    async save(session) {
      cache.set(session.sessionId, session);
      // bound the memory cache: drop the oldest finished sessions beyond 500 (they stay in the database if it is connected)
      if (cache.size > 500 && isDbReady()) {
        for (const [id, s] of cache) {
          if (cache.size <= 500) break;
          if (s.status === 'completed') cache.delete(id);
        }
      }
      if (isDbReady()) {
        await model.findOneAndUpdate({ sessionId: session.sessionId }, { $set: session }, { upsert: true });
        return true;
      }
      return false;
    },
    // All sessions of one user (userId or anonymousId), newest data wins (cache over database).
    async list(key) {
      const byId = new Map();
      if (isDbReady()) {
        const docs = await model.find({ $or: [{ userId: key }, { anonymousId: key }] }).lean();
        for (const d of docs) byId.set(d.sessionId, d);
      }
      for (const s of cache.values()) if (owns(s, key)) byId.set(s.sessionId, s);
      return [...byId.values()];
    },
    async remove(id) {
      const had = cache.delete(id);
      let removed = 0;
      if (isDbReady()) removed = (await model.deleteOne({ sessionId: id })).deletedCount;
      return had || removed > 0;
    },
    // Deletes every session of one user. Returns how many were removed.
    async removeByUser(key) {
      let n = 0;
      for (const [id, s] of [...cache]) {
        if (owns(s, key)) {
          cache.delete(id);
          n += 1;
        }
      }
      if (isDbReady()) n = Math.max(n, (await model.deleteMany({ $or: [{ userId: key }, { anonymousId: key }] })).deletedCount);
      return n;
    },
    persistent: () => isDbReady(),
  };
}
