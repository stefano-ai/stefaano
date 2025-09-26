import { FormEvent, useState } from 'react';
import { useRoomStore } from '../store/roomStore';

export function Lobby() {
  const setJoinDetails = useRoomStore((state) => state.setJoinDetails);
  const [roomId, setRoomId] = useState('demo-room');
  const [role, setRole] = useState<'gm' | 'player'>('gm');
  const [playerId, setPlayerId] = useState('');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!roomId.trim()) return;

    setJoinDetails({
      roomId: roomId.trim(),
      role,
      playerId: playerId.trim() || undefined,
    });
  };

  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 bg-slate-900 p-6 text-slate-100">
      <div className="w-full max-w-md rounded-xl bg-slate-800 p-6 shadow-xl">
        <h1 className="text-3xl font-bold">D&D Tabletop Simulator</h1>
        <p className="mt-2 text-sm text-slate-300">
          Connect as a Game Master or player to collaborate on a shared tabletop.
        </p>
        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <label className="flex flex-col text-sm">
            <span className="mb-1 font-medium">Room ID</span>
            <input
              className="rounded-md border border-slate-700 bg-slate-900 p-2 focus:border-orange-400 focus:outline-none"
              value={roomId}
              onChange={(event) => setRoomId(event.target.value)}
              placeholder="Enter room code"
            />
          </label>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="role"
                value="gm"
                checked={role === 'gm'}
                onChange={() => setRole('gm')}
              />
              Game Master
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="role"
                value="player"
                checked={role === 'player'}
                onChange={() => setRole('player')}
              />
              Player
            </label>
          </div>
          {role === 'player' && (
            <label className="flex flex-col text-sm">
              <span className="mb-1 font-medium">Player ID</span>
              <input
                className="rounded-md border border-slate-700 bg-slate-900 p-2 focus:border-orange-400 focus:outline-none"
                value={playerId}
                onChange={(event) => setPlayerId(event.target.value)}
                placeholder="Who are you?"
                required
              />
            </label>
          )}
          <button
            type="submit"
            className="w-full rounded-md bg-orange-500 py-2 font-semibold text-white transition hover:bg-orange-400"
          >
            Join Table
          </button>
        </form>
      </div>
    </div>
  );
}

export default Lobby;
