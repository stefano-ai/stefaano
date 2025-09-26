import { useEffect, useRef } from 'react';
import { Application, Container, Graphics, InteractionEvent, Text } from 'pixi.js';
import { Token } from '@dnd-tabletop-sim/shared';
import { useRoomStore } from '../store/roomStore';
import { emitTokenUpdate, TableSocket } from '../hooks/useTableSocket';

interface TabletopCanvasProps {
  socket: TableSocket | null;
  isGM: boolean;
  locked: boolean;
}

type DraggableContainer = Container & {
  dragData?: InteractionEvent['data'];
  dragOffset?: { x: number; y: number };
};

const GRID_SIZE = 64;

export function TabletopCanvas({ socket, isGM, locked }: TabletopCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const gridLayerRef = useRef<Graphics | null>(null);
  const tokensLayerRef = useRef<Container | null>(null);
  const joinDetails = useRoomStore((state) => state.joinDetails);
  const tokens = useRoomStore((state) => state.roomState?.tokens ?? []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let appInstance: Application | null = null;

    const drawGrid = () => {
      const gridLayer = gridLayerRef.current;
      const app = appRef.current;
      if (!gridLayer || !app) return;

      const { width, height } = app.renderer.screen;
      gridLayer.clear();
      gridLayer.lineStyle(1, 0x1f2937, 0.7);

      for (let x = 0; x <= width; x += GRID_SIZE) {
        gridLayer.moveTo(x, 0);
        gridLayer.lineTo(x, height);
      }

      for (let y = 0; y <= height; y += GRID_SIZE) {
        gridLayer.moveTo(0, y);
        gridLayer.lineTo(width, y);
      }
    };

    const app = new Application({
      resizeTo: container,
      backgroundColor: 0x0f172a,
      antialias: true,
    });

    appInstance = app;
    appRef.current = app;
    app.stage.sortableChildren = true;
    app.stage.interactive = true;

    const view = app.view as HTMLCanvasElement;
    view.style.width = '100%';
    view.style.height = '100%';
    container.appendChild(view);

    const gridLayer = new Graphics();
    gridLayerRef.current = gridLayer;
    app.stage.addChild(gridLayer);

    const tokensLayer = new Container();
    tokensLayer.sortableChildren = true;
    tokensLayerRef.current = tokensLayer;
    app.stage.addChild(tokensLayer);

    drawGrid();
    window.addEventListener('resize', drawGrid);

    return () => {
      window.removeEventListener('resize', drawGrid);
      if (appInstance) {
        const canvas = appInstance.view as HTMLCanvasElement;
        if (canvas?.parentNode) {
          canvas.parentNode.removeChild(canvas);
        }
        appInstance.destroy(true);
      }
      appRef.current = null;
      gridLayerRef.current = null;
      tokensLayerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const tokensLayer = tokensLayerRef.current;
    const app = appRef.current;
    if (!tokensLayer || !app) return;

    tokensLayer.removeChildren();

    const canDrag = (token: Token) => {
      if (isGM) return true;
      if (locked) return false;
      if (!joinDetails) return false;
      if (!token.ownerId) return true;
      return token.ownerId === joinDetails.playerId;
    };

    tokens.forEach((token) => {
      const tokenContainer = new Container() as DraggableContainer;
      tokenContainer.position.set(token.position.x, token.position.y);
      tokenContainer.zIndex = 10;

      const radius = 24;
      const circle = new Graphics();
      circle.beginFill(token.color ? Number.parseInt(token.color.replace('#', ''), 16) : 0xf97316, 1);
      circle.lineStyle(2, 0xffffff, 0.6);
      circle.drawCircle(0, 0, radius);
      circle.endFill();
      tokenContainer.addChild(circle);

      const label = new Text(token.label, {
        fill: 0xffffff,
        fontFamily: 'Inter, sans-serif',
        fontSize: 12,
        align: 'center',
      });
      label.anchor.set(0.5, -1.4);
      tokenContainer.addChild(label);

      const allowDrag = canDrag(token);
      if (allowDrag) {
        tokenContainer.interactive = true;
        tokenContainer.cursor = 'pointer';
        tokenContainer.buttonMode = true;

        const onPointerDown = (event: InteractionEvent) => {
          tokenContainer.dragData = event.data;
          const localPos = event.data.getLocalPosition(tokensLayer);
          tokenContainer.dragOffset = {
            x: localPos.x - tokenContainer.position.x,
            y: localPos.y - tokenContainer.position.y,
          };
          tokenContainer.alpha = 0.85;
        };

        const onPointerUp = () => {
          if (!tokenContainer.dragData) return;
          tokenContainer.alpha = 1;
          const snappedX = Math.round(tokenContainer.position.x / GRID_SIZE) * GRID_SIZE;
          const snappedY = Math.round(tokenContainer.position.y / GRID_SIZE) * GRID_SIZE;
          tokenContainer.position.set(snappedX, snappedY);
          emitTokenUpdate(socket, {
            tokenId: token.id,
            changes: { position: { x: snappedX, y: snappedY } },
          });
          tokenContainer.dragData = undefined;
          tokenContainer.dragOffset = undefined;
        };

        const onPointerMove = () => {
          if (!tokenContainer.dragData || !tokenContainer.dragOffset) return;
          const newPosition = tokenContainer.dragData.getLocalPosition(tokensLayer);
          tokenContainer.position.set(
            newPosition.x - tokenContainer.dragOffset.x,
            newPosition.y - tokenContainer.dragOffset.y,
          );
        };

        tokenContainer.on('pointerdown', onPointerDown);
        tokenContainer.on('pointerup', onPointerUp);
        tokenContainer.on('pointerupoutside', onPointerUp);
        tokenContainer.on('pointermove', onPointerMove);
      }

      tokensLayer.addChild(tokenContainer);
    });
  }, [tokens, isGM, locked, joinDetails, socket]);

  return <div ref={containerRef} className="flex-1 bg-slate-950" />;
}
