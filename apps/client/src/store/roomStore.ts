import { create } from 'zustand';
import {
  AnnouncementPayload,
  DiceRollResult,
  InitiativeOrderPayload,
  LockStatePayload,
  RoomState,
  Token,
  UserRole,
} from '@dnd-tabletop-sim/shared';

export interface JoinDetails {
  roomId: string;
  role: UserRole;
  playerId?: string;
}

interface RoomStoreState {
  joinDetails?: JoinDetails;
  roomState?: RoomState;
  diceHistory: DiceRollResult[];
  announcements: AnnouncementPayload[];
  setJoinDetails: (details: JoinDetails) => void;
  clearJoin: () => void;
  setRoomState: (state: RoomState) => void;
  upsertToken: (token: Token) => void;
  addDiceResult: (result: DiceRollResult) => void;
  addAnnouncement: (announcement: AnnouncementPayload) => void;
  setLockState: (payload: LockStatePayload) => void;
  setInitiativeOrder: (payload: InitiativeOrderPayload) => void;
}

export const useRoomStore = create<RoomStoreState>((set) => ({
  diceHistory: [],
  announcements: [],
  setJoinDetails: (details) => set({ joinDetails: details }),
  clearJoin: () => set({ joinDetails: undefined, roomState: undefined, diceHistory: [], announcements: [] }),
  setRoomState: (state) =>
    set({
      roomState: state,
      diceHistory: state.diceHistory,
    }),
  upsertToken: (token) =>
    set((current) => {
      const roomState = current.roomState;
      if (!roomState) return {};
      const tokens = roomState.tokens.filter((existing) => existing.id !== token.id);
      tokens.push(token);
      return { roomState: { ...roomState, tokens } };
    }),
  addDiceResult: (result) =>
    set((current) => ({
      diceHistory: [...current.diceHistory, result].slice(-50),
      roomState: current.roomState
        ? { ...current.roomState, diceHistory: [...current.diceHistory, result].slice(-50) }
        : current.roomState,
    })),
  addAnnouncement: (announcement) =>
    set((current) => ({
      announcements: [...current.announcements, announcement].slice(-50),
    })),
  setLockState: (payload) =>
    set((current) => ({
      roomState: current.roomState ? { ...current.roomState, locked: payload.locked } : current.roomState,
    })),
  setInitiativeOrder: (payload) =>
    set((current) => ({
      roomState: current.roomState
        ? { ...current.roomState, initiativeOrder: [...payload.order] }
        : current.roomState,
    })),
}));
