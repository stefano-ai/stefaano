export type UserRole = 'gm' | 'player';

export interface JoinRoomPayload {
  roomId: string;
  role: UserRole;
  playerId?: string;
}

export interface TokenPosition {
  x: number;
  y: number;
}

export type TokenKind = 'player' | 'npc';

export interface Token {
  id: string;
  label: string;
  kind: TokenKind;
  position: TokenPosition;
  ownerId?: string;
  color?: string;
}

export interface DiceRollRequest {
  formula: string;
  rollerId?: string;
  context?: string;
  advantage?: boolean;
  disadvantage?: boolean;
}

export interface DiceRollResult extends DiceRollRequest {
  total: number;
  rolls: number[];
  detail: string;
  timestamp: string;
}

export interface AnnouncementPayload {
  message: string;
  authorRole: UserRole;
  timestamp: string;
}

export interface RoomState {
  roomId: string;
  tokens: Token[];
  initiativeOrder: string[];
  diceHistory: DiceRollResult[];
  locked: boolean;
}

export interface LockStatePayload {
  locked: boolean;
}

export interface TokenAddPayload {
  token: Token;
}

export interface TokenUpdatePayload {
  tokenId: string;
  changes: Partial<Omit<Token, 'id'>>;
}

export interface InitiativeOrderPayload {
  order: string[];
}

export interface SocketToServerEvents {
  join_room: (payload: JoinRoomPayload) => void;
  token_add: (payload: TokenAddPayload) => void;
  token_update: (payload: TokenUpdatePayload) => void;
  dice_roll: (payload: DiceRollRequest) => void;
  announce: (payload: AnnouncementPayload) => void;
  initiative_update: (payload: InitiativeOrderPayload) => void;
  lock_update: (payload: LockStatePayload) => void;
}

export interface SocketToClientEvents {
  room_state: (payload: RoomState) => void;
  token_added: (payload: TokenAddPayload) => void;
  token_updated: (payload: Token) => void;
  dice_result: (payload: DiceRollResult) => void;
  announcement: (payload: AnnouncementPayload) => void;
  initiative_order: (payload: InitiativeOrderPayload) => void;
  lock_state: (payload: LockStatePayload) => void;
}

export interface InterServerEvents {}

export interface SocketData {
  roomId?: string;
  role?: UserRole;
  playerId?: string;
}
