/* ═══════════════════════════════════════════════════════════════
   AEROS — WebSocket Client Manager
   Handles reconnect, heartbeat ping, and dispatch of server messages.
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  class LiveSocket {
    constructor(url) {
      this.url = url;
      this.socket = null;
      this.handlers = new Set();
      this.status = 'connecting';
      this._retries = 0;
      this._reconnectTimer = null;
      this._statusEl = null;
    }

    setStatusEl(el) {
      this._statusEl = el;
    }

    connect() {
      try {
        this.socket = new WebSocket(this.url);
        this.setStatus('connecting');
      } catch (e) {
        this.scheduleReconnect();
        return;
      }

      this.socket.onopen = () => {
        this.setStatus('connected');
        this._retries = 0;
      };

      this.socket.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'heartbeat') {
            this.socket.send('ping');
            return;
          }
          this.handlers.forEach((h) => {
            try { h(msg); } catch (e) { console.error('WS handler', e); }
          });
        } catch (e) {
          console.warn('Invalid WS message', e);
        }
      };

      this.socket.onclose = () => {
        this.setStatus('stale');
        this.scheduleReconnect();
      };

      this.socket.onerror = () => {
        this.setStatus('stale');
      };
    }

    onMessage(handler) {
      this.handlers.add(handler);
      return () => this.handlers.delete(handler);
    }

    send(obj) {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(typeof obj === 'string' ? obj : JSON.stringify(obj));
      }
    }

    setStatus(s) {
      this.status = s;
      if (this._statusEl) {
        this._statusEl.className = 'ws-status ' + s;
      }
    }

    scheduleReconnect() {
      if (this._reconnectTimer) return;
      const delay = Math.min(30000, 1000 * Math.pow(2, this._retries++));
      this._reconnectTimer = setTimeout(() => {
        this._reconnectTimer = null;
        this.connect();
      }, delay);
    }

    close() {
      if (this.socket) this.socket.close();
      if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    }
  }

  global.LiveSocket = LiveSocket;
})(window);