import express from 'express';
import http from 'http';
import cors from 'cors';
import { Server } from 'socket.io';
import {
  AnnouncementPayload,
  DiceRollRequest,
  InitiativeOrderPayload,
  LockStatePayload,
  SocketData,
  SocketToClientEvents,
  SocketToServerEvents,
  TokenAddPayload,
  TokenUpdatePayload,
} from '@dnd-tabletop-sim/shared';
import { RoomManager } from './roomManager';
import { rollDice } from './dice';

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);

const io = new Server<SocketToServerEvents, SocketToClientEvents, Record<string, never>, SocketData>(
  server,
  {
    cors: {
      origin: '*',
      methods: ['GET', 'POST'],
    },
  },
);

const TABLE_NAMESPACE = '/table';
const roomManager = new RoomManager();

io.of(TABLE_NAMESPACE).on('connection', (socket) => {
  socket.on('join_room', (payload) => {
    const { roomId, role, playerId } = payload;
    socket.data.roomId = roomId;
    socket.data.role = role;
    socket.data.playerId = playerId;
    socket.join(roomId);

    const state = roomManager.getState(roomId);
    socket.emit('room_state', state);
  });

  socket.on('token_add', (payload: TokenAddPayload) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    if (socket.data.role !== 'gm') {
      return;
    }

    const token = roomManager.addToken(roomId, payload);
    io.of(TABLE_NAMESPACE).to(roomId).emit('token_added', { token });
    const state = roomManager.getState(roomId);
    io.of(TABLE_NAMESPACE).to(roomId).emit('room_state', state);
  });

  socket.on('token_update', (payload: TokenUpdatePayload) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    const state = roomManager.getState(roomId);
    if (state.locked && socket.data.role !== 'gm') {
      return;
    }

    const updatedToken = roomManager.updateToken(roomId, payload);
    if (!updatedToken) {
      return;
    }

    io.of(TABLE_NAMESPACE).to(roomId).emit('token_updated', updatedToken);
  });

  socket.on('dice_roll', (payload: DiceRollRequest) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    try {
      const result = rollDice(payload);
      roomManager.recordDiceRoll(roomId, payload, result);
      io.of(TABLE_NAMESPACE).to(roomId).emit('dice_result', result);
    } catch (error) {
      if (error instanceof Error) {
        const announcement: AnnouncementPayload = {
          message: `Dice roll failed: ${error.message}`,
          authorRole: 'gm',
          timestamp: new Date().toISOString(),
        };
        socket.emit('announcement', announcement);
      }
    }
  });

  socket.on('announce', (payload: AnnouncementPayload) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    const announcement = roomManager.addAnnouncement(roomId, payload);
    io.of(TABLE_NAMESPACE).to(roomId).emit('announcement', announcement);
  });

  socket.on('initiative_update', (payload: InitiativeOrderPayload) => {
    const roomId = socket.data.roomId;
    if (!roomId || socket.data.role !== 'gm') return;

    const updated = roomManager.setInitiative(roomId, payload);
    io.of(TABLE_NAMESPACE).to(roomId).emit('initiative_order', updated);
  });

  socket.on('lock_update', (payload: LockStatePayload) => {
    const roomId = socket.data.roomId;
    if (!roomId || socket.data.role !== 'gm') return;

    const updated = roomManager.toggleLock(roomId, payload.locked);
    io.of(TABLE_NAMESPACE).to(roomId).emit('lock_state', updated);
    const state = roomManager.getState(roomId);
    io.of(TABLE_NAMESPACE).to(roomId).emit('room_state', state);
  });
});

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT ? Number.parseInt(process.env.PORT, 10) : 4000;
server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
