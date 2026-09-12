/* ═══════════════════════════════════════════════════════════════
   AEROS — MapLibre GL Map Controller
   Base map + spatial AQI layers (heatmap, stations, plumes, fires).
   ═══════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  const DELHI_CENTER = [77.2090, 28.6139];
  const CARTO_TILES = ['https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'];
  const CARTO_ATTR = '© CARTO';

  function maptilerTiles(key) {
    return [`https://api.maptiler.com/maps/darkmatter/{z}/{x}/{y}.png?key=${key}`];
  }
  const FIRE_REGIONS = [
    { name: 'Punjab', colors: '#ff3838', lat: 30.8, lon: 75.4 },
    { name: 'Haryana', colors: '#ffb800', lat: 29.6, lon: 76.4 },
  ];

  class AeriMap {
    constructor(containerId, maptilerKey) {
      this.container = document.getElementById(containerId);
      this.onStationClick = null;
      this.currentHour = 0;
      this._forecasts = {};
      // Better alternative: use the premium MapTiler dark basemap when a
      // key is configured, otherwise fall back to free CARTO tiles.
      const tiles = maptilerKey ? maptilerTiles(maptilerKey) : CARTO_TILES;
      const attribution = maptilerKey
        ? '© MapTiler © OpenStreetMap contributors'
        : CARTO_ATTR;
      this.map = new maplibregl.Map({
        container: this.container,
        style: {
          version: 8,
          sources: {
            base: {
              type: 'raster',
              tiles: tiles,
              tileSize: 256,
              attribution: attribution,
            },
          },
          layers: [{
            id: 'base',
            type: 'raster',
            source: 'base',
          }],
        },
        center: DELHI_CENTER,
        zoom: 8.2,
        attributionControl: true,
      });

      this.map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');
      this.map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');

      this.map.on('load', () => this._onLoad());
    }

    _onLoad() {
      this._initSources();
      this._bindEvents();
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = 'map ready · live spatial overlay';
    }

    _initSources() {
      // PM2.5 heatmap source (station points with pm25 weight)
      this.map.addSource('pm25-heat', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'pm25-heat',
        type: 'heatmap',
        source: 'pm25-heat',
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'pm25'], 0, 0, 150, 0.6, 500, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 2.4],
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0, 'rgba(0,0,0,0)',
            0.15, 'rgba(0,212,255,0.25)',
            0.35, 'rgba(255,255,0,0.35)',
            0.5, 'rgba(255,126,0,0.55)',
            0.7, 'rgba(255,0,0,0.75)',
            0.9, 'rgba(153,0,76,0.9)',
            1, 'rgba(126,0,35,0.95)',
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 12, 9, 34],
          'heatmap-opacity': 0.75,
        },
      });

      // Stations (circles colored by AQI)
      this.map.addSource('stations', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'stations-glow',
        type: 'circle',
        source: 'stations',
        paint: {
          'circle-radius': 12,
          'circle-color': ['get', 'color'],
          'circle-opacity': 0.25,
          'circle-blur': 1,
        },
      });
      this.map.addLayer({
        id: 'stations',
        type: 'circle',
        source: 'stations',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 5, 10, 8],
          'circle-color': ['get', 'color'],
          'circle-stroke-color': 'rgba(255,255,255,0.85)',
          'circle-stroke-width': 1.4,
        },
      });

      // Fire hotspots (pulsing red)
      this.map.addSource('fires', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'fires-halo',
        type: 'circle',
        source: 'fires',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'frp'], 0, 6, 80, 16],
          'circle-color': '#ff3838',
          'circle-opacity': 0.35,
          'circle-blur': 0.8,
        },
      });
      this.map.addLayer({
        id: 'fires',
        type: 'circle',
        source: 'fires',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'frp'], 0, 3, 80, 11],
          'circle-color': '#ff3838',
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1,
        },
      });

      // Plume trajectories
      this.map.addSource('plumes', { type: 'geojson', data: emptyFC() });
      this.map.addLayer({
        id: 'plumes-dash',
        type: 'line',
        source: 'plumes',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'estimated_contribution_pm25'], 0, '#39ff14', 5, '#ffb800', 15, '#ff3838'],
          'line-width': 2.2,
          'line-opacity': 0.85,
          'line-dasharray': [2, 1.4],
        },
      });

      // Delhi domain ring approximate boundary
      this.map.addSource('delhi-ring', {
        type: 'geojson',
        data: this._delhiRingGeoJSON(),
      });
      this.map.addLayer({
        id: 'delhi-ring',
        type: 'line',
        source: 'delhi-ring',
        paint: {
          'line-color': 'rgba(0,212,255,0.5)',
          'line-width': 1.2,
          'line-dasharray': [1, 1],
        },
      });
    }

    _bindEvents() {
      this.map.on('click', 'stations', (e) => {
        const prop = (e.features[0] || {}).properties || {};
        if (this.onStationClick) this.onStationClick(prop.id || prop.name);
        this._flyTo(e.lngLat.lng, e.lngLat.lat);
      });

      this.map.on('mouseenter', 'stations', () => {
        this.map.getCanvas().style.cursor = 'pointer';
      });
      this.map.on('mouseleave', 'stations', () => {
        this.map.getCanvas().style.cursor = '';
      });

      this.map.on('mousemove', 'fires', (e) => {
        const p = e.features[0].properties;
        this._tooltip(`Fire @ ${p.latitude.toFixed?.(2) ?? p.lat ?? ''} · FRP ${p.frp} MW`);
      });
      this.map.on('mouseleave', 'fires', () => this._clearTooltip());
    }

    _flyTo(lon, lat) {
      this.map.flyTo({ center: [lon, lat], zoom: 9.5, duration: 800 });
    }

    _tooltip(text) {
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = text;
    }
    _clearTooltip() {
      const el = document.getElementById('mapStatus');
      if (el) el.textContent = 'hover a fire marker for details';
    }

    /* ── Public update API ─────────────────────────────────────── */
    update(forecastGeojson, stationsGeojson, firesGeojson, plumesGeojson) {
      if (!this.map.isStyleLoaded()) return;
      this._updateSource('stations', stationsGeojson || emptyFC());
      this._updateSource('pm25-heat', forecastGeojson || stationsGeojson || emptyFC());
      this._updateSource('fires', firesGeojson || emptyFC());
      this._updateSource('plumes', plumesGeojson || emptyFC());
    }

    setTime(hour) {
      this.currentHour = hour;
      // Stations layer can reflect forecast hour if forecasts provided
      const fc = this._forecasts;
      if (!fc || !Object.keys(fc).length) return;
      const features = [];
      for (const id in fc) {
        const f = fc[id];
        if (!f || !f.pm25 || hour >= f.pm25.length) continue;
        const st = this._stationLookup && this._stationLookup[id];
        if (!st) continue;
        features.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [st.longitude, st.latitude] },
          properties: {
            id: id,
            name: st.short_name,
            pm25: f.pm25[hour],
            aqi: (f.aqi || [])[hour] || 0,
            category: (f.category || [])[hour] || 'Unknown',
            color: (f.colors || [])[hour] || '#808080',
          },
        });
      }
      this._updateSource('stations', { type: 'FeatureCollection', features });
      this._updateSource('pm25-heat', { type: 'FeatureCollection', features });
    }

    setForecasts(forecasts, stationLookup) {
      this._forecasts = forecasts || {};
      this._stationLookup = stationLookup || null;
    }

    toggleLayer(id, visible) {
      const layerIds = {
        heat: 'pm25-heat',
        fires: ['fires', 'fires-halo'],
        plumes: 'plumes-dash',
        stations: ['stations', 'stations-glow'],
      };
      const targets = layerIds[id];
      if (!targets) return;
      const args = Array.isArray(targets) ? targets : [targets];
      args.forEach((lid) => {
        if (this.map.getLayer(lid)) this.map.setLayoutProperty(lid, 'visibility', visible ? 'visible' : 'none');
      });
    }

    /* ── Helpers ───────────────────────────────────────────────── */
    _updateSource(id, data) {
      const src = this.map.getSource(id);
      if (src) src.setData(data);
    }

    _delhiRingGeoJSON() {
      const features = [];
      const R = 0.75; // deg approx
      const coords = [];
      for (let i = 0; i <= 48; i++) {
        const a = (i / 48) * Math.PI * 2;
        coords.push([
          DELHI_CENTER[0] + R * Math.cos(a) * 1.15,
          DELHI_CENTER[1] + R * Math.sin(a) * 0.92,
        ]);
      }
      features.push({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords },
        properties: { name: 'Delhi NCR approx boundary' },
      });
      return { type: 'FeatureCollection', features };
    }
  }

  function emptyFC() {
    return { type: 'FeatureCollection', features: [] };
  }

  global.AeriMap = AeriMap;
  global.DELHI_CENTER = DELHI_CENTER;
})(window);