import {
  AnnouncementPayload,
  DiceRollRequest,
  DiceRollResult,
  InitiativeOrderPayload,
  LockStatePayload,
  RoomState,
  Token,
  TokenAddPayload,
  TokenUpdatePayload,
} from '@dnd-tabletop-sim/shared';

type RoomInternalState = Omit<RoomState, 'roomId'>;

export class RoomManager {
  private rooms = new Map<string, RoomInternalState>();

  getState(roomId: string): RoomState {
    const state = this.ensureRoom(roomId);
    return { roomId, ...state };
  }

  addToken(roomId: string, payload: TokenAddPayload): Token {
    const state = this.ensureRoom(roomId);
    state.tokens = state.tokens.filter((token) => token.id !== payload.token.id);
    state.tokens.push(payload.token);
    return payload.token;
  }

  updateToken(roomId: string, payload: TokenUpdatePayload): Token | undefined {
    const state = this.ensureRoom(roomId);
    const token = state.tokens.find((item) => item.id === payload.tokenId);
    if (!token) {
      return undefined;
    }

    Object.assign(token, payload.changes);
    return token;
  }

  setInitiative(roomId: string, payload: InitiativeOrderPayload): InitiativeOrderPayload {
    const state = this.ensureRoom(roomId);
    state.initiativeOrder = [...payload.order];
    return { order: [...state.initiativeOrder] };
  }

  recordDiceRoll(roomId: string, request: DiceRollRequest, result: DiceRollResult): DiceRollResult {
    const state = this.ensureRoom(roomId);
    state.diceHistory = [...state.diceHistory, result].slice(-50);
    return result;
  }

  addAnnouncement(roomId: string, announcement: AnnouncementPayload): AnnouncementPayload {
    this.ensureRoom(roomId);
    // Could store announcements if needed later.
    return announcement;
  }

  toggleLock(roomId: string, locked: boolean): LockStatePayload {
    const state = this.ensureRoom(roomId);
    state.locked = locked;
    return { locked: state.locked };
  }

  private ensureRoom(roomId: string): RoomInternalState {
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, {
        tokens: [],
        initiativeOrder: [],
        diceHistory: [],
        locked: false,
      });
    }

    return this.rooms.get(roomId)!;
  }
}
