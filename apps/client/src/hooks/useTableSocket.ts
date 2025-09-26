import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import {
  SocketToClientEvents,
  SocketToServerEvents,
  TokenAddPayload,
  TokenUpdatePayload,
  DiceRollRequest,
  AnnouncementPayload,
  InitiativeOrderPayload,
  LockStatePayload,
} from '@dnd-tabletop-sim/shared';
import { useRoomStore } from '../store/roomStore';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL ?? 'http://localhost:4000';
const TABLE_NAMESPACE = '/table';

export type TableSocket = Socket<SocketToClientEvents, SocketToServerEvents>;

export function useTableSocket(): TableSocket | null {
  const joinDetails = useRoomStore((state) => state.joinDetails);
  const setRoomState = useRoomStore((state) => state.setRoomState);
  const upsertToken = useRoomStore((state) => state.upsertToken);
  const addDiceResult = useRoomStore((state) => state.addDiceResult);
  const addAnnouncement = useRoomStore((state) => state.addAnnouncement);
  const setLockState = useRoomStore((state) => state.setLockState);
  const setInitiativeOrder = useRoomStore((state) => state.setInitiativeOrder);

  const socketRef = useRef<TableSocket | null>(null);
  const [socketState, setSocketState] = useState<TableSocket | null>(null);

  useEffect(() => {
    if (!joinDetails) {
      socketRef.current?.disconnect();
      socketRef.current = null;
      setSocketState(null);
      return;
    }

    const socket = io(`${SOCKET_URL}${TABLE_NAMESPACE}`, {
      transports: ['websocket'],
    }) as TableSocket;
    socketRef.current = socket;
    setSocketState(socket);

    socket.emit('join_room', joinDetails);

    socket.on('room_state', setRoomState);
    socket.on('token_added', (payload) => upsertToken(payload.token));
    socket.on('token_updated', upsertToken);
    socket.on('dice_result', addDiceResult);
    socket.on('announcement', addAnnouncement);
    socket.on('initiative_order', setInitiativeOrder);
    socket.on('lock_state', setLockState);

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
      socketRef.current = null;
      setSocketState(null);
    };
  }, [joinDetails, setRoomState, upsertToken, addDiceResult, addAnnouncement, setLockState, setInitiativeOrder]);

  return socketState;
}

export function emitTokenAdd(socket: TableSocket | null, payload: TokenAddPayload) {
  socket?.emit('token_add', payload);
}

export function emitTokenUpdate(socket: TableSocket | null, payload: TokenUpdatePayload) {
  socket?.emit('token_update', payload);
}

export function emitDiceRoll(socket: TableSocket | null, payload: DiceRollRequest) {
  socket?.emit('dice_roll', payload);
}

export function emitAnnouncement(socket: TableSocket | null, payload: AnnouncementPayload) {
  socket?.emit('announce', payload);
}

export function emitInitiativeUpdate(socket: TableSocket | null, payload: InitiativeOrderPayload) {
  socket?.emit('initiative_update', payload);
}

export function emitLockUpdate(socket: TableSocket | null, payload: LockStatePayload) {
  socket?.emit('lock_update', payload);
}
