import { FormEvent, useState } from 'react';
import { AnnouncementPayload } from '@dnd-tabletop-sim/shared';

interface AnnouncementFeedProps {
  announcements: AnnouncementPayload[];
  onAnnounce: (payload: AnnouncementPayload) => void;
  isGM: boolean;
}

export function AnnouncementFeed({ announcements, onAnnounce, isGM }: AnnouncementFeedProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!message.trim()) return;

    onAnnounce({
      message: message.trim(),
      authorRole: isGM ? 'gm' : 'player',
      timestamp: new Date().toISOString(),
    });
    setMessage('');
  };

  return (
    <section className="flex h-full flex-col">
      <header className="border-b border-slate-800 p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Announcements</h3>
      </header>
      <div className="flex-1 overflow-y-auto p-4">
        <ul className="space-y-3 text-sm">
          {announcements.length === 0 && <li className="text-xs text-slate-500">No announcements yet.</li>}
          {announcements
            .slice()
            .reverse()
            .map((announcement) => (
              <li key={announcement.timestamp} className="rounded-md bg-slate-900/60 p-3">
                <p className="font-medium text-slate-100">{announcement.message}</p>
                <p className="text-[11px] uppercase tracking-wide text-slate-500">
                  {announcement.authorRole.toUpperCase()} •
                  {new Date(announcement.timestamp).toLocaleTimeString()}
                </p>
              </li>
            ))}
        </ul>
      </div>
      <footer className="border-t border-slate-800 p-4">
        <form className="space-y-2" onSubmit={handleSubmit}>
          <textarea
            className="h-20 w-full rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:border-orange-400 focus:outline-none"
            placeholder="Share updates with the table"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            disabled={!isGM}
          />
          <button
            type="submit"
            className="w-full rounded-md bg-orange-500 py-2 text-sm font-semibold text-white transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-slate-700"
            disabled={!isGM}
          >
            Post Announcement
          </button>
        </form>
      </footer>
    </section>
  );
}
