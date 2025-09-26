import { useRoomStore } from './store/roomStore';
import { Lobby } from './components/Lobby';
import { useTableSocket } from './hooks/useTableSocket';
import { TabletopView } from './components/TabletopView';

function App() {
  const joinDetails = useRoomStore((state) => state.joinDetails);
  const socket = useTableSocket();

  if (!joinDetails) {
    return <Lobby />;
  }

  return <TabletopView socket={socket} />;
}

export default App;
