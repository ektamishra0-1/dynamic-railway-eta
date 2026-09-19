import { useEffect, useMemo, useRef, useState } from "react";

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bell,
  Brain,
  ChevronDown,
  Clock3,
  Gauge,
  MapPin,
  MessageCircle,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Route,
  Send,
  Share2,
  ShieldCheck,
  Sparkles,
  Ticket,
  TrainFront,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./App.css";


const API = "http://127.0.0.1:8000";


function App() {

  const [trains, setTrains] = useState([]);
  const [trainNo, setTrainNo] = useState("");
  const [journeyDate, setJourneyDate] = useState("");
  const [journeyDates, setJourneyDates] = useState([]);

  const [route, setRoute] = useState([]);
  const [liveState, setLiveState] = useState(null);
  const [forecast, setForecast] = useState(null);

  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedStation, setSelectedStation] = useState(null);
  const [activeTab, setActiveTab] = useState("journey");

  const [assistantOpen, setAssistantOpen] = useState(false);
  const [whatIfOpen, setWhatIfOpen] = useState(false);
  const [alertsOpen, setAlertsOpen] = useState(false);

  const [assistantMessage, setAssistantMessage] = useState("");
  const [assistantReply, setAssistantReply] = useState("");

  const [whatIfDelay, setWhatIfDelay] = useState(20);

  const updating = useRef(false);


  // ======================================================
  // LOAD TRAINS
  // ======================================================

  useEffect(() => {

    async function loadTrains() {

      try {

        const response = await fetch(
          `${API}/trains`
        );

        if (!response.ok) {
          throw new Error();
        }

        const data = await response.json();

        const list = data.trains || [];

        setTrains(list);

        if (list.length) {
          setTrainNo(list[0]);
        }

      } catch (err) {

        console.error(err);

        setError(
          "Backend connection failed. Start FastAPI on port 8000."
        );

      } finally {

        setLoading(false);

      }

    }

    loadTrains();

  }, []);


  // ======================================================
  // LOAD JOURNEY DATES
  // ======================================================

  useEffect(() => {

    if (!trainNo) return;

    async function loadDates() {

      try {

        const response = await fetch(
          `${API}/journeys/${trainNo}`
        );

        if (!response.ok) {
          throw new Error();
        }

        const data = await response.json();

        const dates = data.dates || [];

        setJourneyDates(dates);

        if (dates.length) {

          setJourneyDate(
            dates.includes("2026-09-04")
              ? "2026-09-04"
              : dates[0]
          );

        }

      } catch (err) {

        console.error(err);

      }

    }

    loadDates();

  }, [trainNo]);


  // ======================================================
  // LOAD FULL ROUTE
  // ======================================================

  async function loadRoute() {

    if (!trainNo || !journeyDate) return;

    try {

      const response = await fetch(
        `${API}/route/${trainNo}/${journeyDate}`
      );

      if (!response.ok) {
        throw new Error("Route unavailable");
      }

      const data = await response.json();

      setRoute(
        data.stations || []
      );

    } catch (err) {

      console.error(err);

      setRoute([]);

    }

  }


  // ======================================================
  // ======================================================
  // STATE NORMALIZER
  // ======================================================

  function extractLiveState(data) {
    if (!data) return null;
    const s = data.state ? data.state : data;
    return {
      ...s,
      station_code: s.station_code || "",
      station_name: s.station_name || "",
      delay: Number.isFinite(Number(s.delay)) ? Number(s.delay) : 0,
      previous_delay: Number.isFinite(Number(s.previous_delay)) ? Number(s.previous_delay) : 0,
      delay_change: Number.isFinite(Number(s.delay_change)) ? Number(s.delay_change) : 0,
      speed_kmph: Number.isFinite(Number(s.speed_kmph)) ? Number(s.speed_kmph) : 72,
      progress: Number.isFinite(Number(s.progress)) ? Number(s.progress) : 0,
      journey_station_index: Number.isFinite(Number(s.journey_station_index)) ? Number(s.journey_station_index) : 1,
      total_stations: Number.isFinite(Number(s.total_stations)) ? Number(s.total_stations) : 7,
      is_finished: Boolean(s.is_finished),
    };
  }


  // ======================================================
  // START TRAIN
  // ======================================================

  async function startLive() {

    if (!trainNo || !journeyDate) return;

    try {

      setError("");

      const response = await fetch(
        `${API}/live/start/${trainNo}/${journeyDate}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(
          "Could not start train replay."
        );
      }

      const data = await response.json();
      const state = extractLiveState(data);

      setLiveState(state);

      if (state?.station_code) {
        setSelectedStation(
          state.station_code
        );
      }

      await loadRoute();
      await loadLiveForecast();

      setRunning(true);

    } catch (err) {

      console.error(err);

      setError(
        err.message
      );

    }

  }


  // ======================================================
  // LIVE TICK
  // ======================================================

  async function updateLive() {

    if (
      updating.current ||
      !trainNo ||
      !journeyDate
    ) {
      return;
    }

    updating.current = true;

    try {

      const response = await fetch(
        `${API}/live/tick/${trainNo}/${journeyDate}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(
          "Live update failed."
        );
      }

      const data = await response.json();
      const state = extractLiveState(data);

      setLiveState(state);

      if (state?.station_code) {
        setSelectedStation(
          state.station_code
        );
      }

      await loadLiveForecast();

      if (state?.is_finished) {
        setRunning(false);
      }

    } catch (err) {

      console.error(err);

      setError(
        err.message
      );

      setRunning(false);

    } finally {

      updating.current = false;

    }

  }


  // ======================================================
  // INITIAL LIVE STATE
  // ======================================================

  async function loadInitialLiveState() {

    if (!trainNo || !journeyDate) return;

    try {

      let response = await fetch(
        `${API}/live/state/${trainNo}/${journeyDate}`
      );

      if (!response.ok) {
        response = await fetch(
          `${API}/live/start/${trainNo}/${journeyDate}`,
          { method: "POST" }
        );
      }

      if (response.ok) {
        const data = await response.json();
        const state = extractLiveState(data);

        setLiveState(state);

        if (state?.station_code) {
          setSelectedStation(
            state.station_code
          );
        }

        await loadLiveForecast();
      }

    } catch (err) {

      console.error("Initial live state error:", err);

    }

  }


  // ======================================================
  // FORECAST
  // ======================================================

  async function loadLiveForecast() {

    if (!trainNo || !journeyDate) return;

    try {

      const response = await fetch(
        `${API}/live/forecast/${trainNo}/${journeyDate}?horizon=4`
      );

      if (!response.ok) {
        throw new Error();
      }

      const data = await response.json();

      setForecast(data);

    } catch (err) {

      /*
       * Fallback to the normal forecast endpoint.
       */

      try {

        const response = await fetch(
          `${API}/forecast/${trainNo}/${journeyDate}?horizon=4`
        );

        if (!response.ok) {
          return;
        }

        const data = await response.json();

        setForecast(data);

      } catch (fallbackError) {

        console.error(
          fallbackError
        );

      }

    }

  }


  // ======================================================
  // AUTO UPDATE
  // ======================================================

  useEffect(() => {

    if (!running) return;

    const timer = setInterval(
      updateLive,
      3000
    );

    return () => {
      clearInterval(timer);
    };

  }, [
    running,
    trainNo,
    journeyDate,
  ]);


  // ======================================================
  // LOAD INITIAL ROUTE & STATE
  // ======================================================

  useEffect(() => {

    if (
      trainNo &&
      journeyDate
    ) {
      loadRoute();
      loadInitialLiveState();
    }

  }, [
    trainNo,
    journeyDate,
  ]);


  // ======================================================
  // GRAPH DATA
  // ======================================================

  const graphData = useMemo(() => {

    if (!forecast && !liveState) {
      return [];
    }

    const current = liveState && liveState.station_code
      ? [
          {
            station:
              liveState.station_code,

            stationName:
              liveState.station_name ||
              liveState.station_code,

            delay: Number.isFinite(Number(liveState.delay))
              ? Number(liveState.delay)
              : 0,

            type: "current",
          },
        ]
      : [];

    const future =
      forecast?.stations?.map(
        (station) => ({
          station:
            station.station_code,

          stationName:
            station.station_name ||
            station.station_code,

          delay: Number.isFinite(Number(station.predicted_delay))
            ? Number(station.predicted_delay)
            : 0,

          lower: Number.isFinite(Number(station.lower_delay))
            ? Number(station.lower_delay)
            : 0,

          upper: Number.isFinite(Number(station.upper_delay))
            ? Number(station.upper_delay)
            : 0,

          confidence: Number.isFinite(Number(station.confidence))
            ? Number(station.confidence)
            : 0,

          type: "forecast",
        })
      ) || [];

    return [
      ...current,
      ...future,
    ];

  }, [
    forecast,
    liveState,
  ]);


  // ======================================================
  // PROGRESS
  // ======================================================

  const currentIndex = Number(
    liveState?.journey_station_index ?? (route[0]?.journey_station_index ?? 1)
  );

  const totalStationsCount = Number(
    liveState?.total_stations || route.length || 7
  );

  const progress = Number(
    liveState?.progress !== undefined && Number.isFinite(Number(liveState.progress))
      ? liveState.progress
      : route.length > 1
        ? (Math.max(0, currentIndex - 1) / Math.max(1, route.length - 1)) * 100
        : 0
  );


  // ======================================================
  // SELECTED STATION
  // ======================================================

  const selectedStationData =
    route.find(
      (station) =>
        station.station_code ===
        selectedStation
    );


  const selectedForecast =
    forecast?.stations?.find(
      (station) =>
        station.station_code ===
        selectedStation
    );


  // ======================================================
  // FORECAST HELPERS
  // ======================================================

  const maxForecastDelay =
    forecast?.stations?.length
      ? Math.max(
          ...forecast.stations.map(
            (s) =>
              Number(
                s.predicted_delay || 0
              )
          )
        )
      : null;


  const averageConfidence =
    forecast?.stations?.length
      ? (
          forecast.stations.reduce(
            (sum, station) =>
              sum +
              Number(
                station.confidence || 0
              ),
            0
          ) /
          forecast.stations.length
        ) * 100
      : null;


  const delayTrend =
    graphData.length >= 2
      ? graphData[
          graphData.length - 1
        ].delay -
        graphData[0].delay
      : 0;


  // ======================================================
  // TRAIN CHANGE
  // ======================================================

  function changeTrain(value) {

    setTrainNo(value);
    setRunning(false);
    setLiveState(null);
    setForecast(null);
    setSelectedStation(null);

  }


  function changeDate(value) {

    setJourneyDate(value);
    setRunning(false);
    setLiveState(null);
    setForecast(null);
    setSelectedStation(null);

  }


  // ======================================================
  // ASSISTANT
  // ======================================================

  function askAssistant() {

    if (!assistantMessage.trim()) return;

    const message =
      assistantMessage.toLowerCase();

    let reply =
      "RailPulse AI is monitoring the train's current state and downstream delay propagation.";

    if (
      message.includes("delay")
    ) {

      reply =
        `Current delay is ${
          Math.round(
            liveState?.delay || 0
          )
        } minutes. The GRU + Attention AI model is forecasting downstream propagation across the next ${forecast?.stations?.length || 4} stations.`;

    } else if (
      message.includes("why")
    ) {

      reply =
        "The prediction is influenced by the current observed delay, previous delay behaviour, station position, train identity and historical temporal patterns.";

    } else if (
      message.includes("next")
    ) {

      const next =
        forecast?.stations?.[0];

      reply = next
        ? `The next forecast station is ${next.station_name || next.station_code}, with an estimated delay of ${Math.round(next.predicted_delay)} minutes.`
        : "Start the train replay first so I can calculate the next-station forecast.";

    }

    setAssistantReply(
      reply
    );

  }


  // ======================================================
  // WHAT IF
  // ======================================================

  const whatIfResult =
    Math.max(
      0,
      Number(
        liveState?.delay || 0
      ) +
      Number(
        whatIfDelay
      )
    );


  // ======================================================
  // LOADING
  // ======================================================

  if (loading) {

    return (

      <div className="app-loading">

        <div className="loading-train">
          <TrainFront size={44} />
        </div>

        <h2>
          Initialising RailPulse AI
        </h2>

        <p>
          Loading railway intelligence engine...
        </p>

      </div>

    );

  }


  return (

    <div className="app">

      {/* ==================================================
          HEADER
      ================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-icon">
            <TrainFront size={23} />
          </div>

          <div>

            <div className="brand-name">
              RailPulse AI
            </div>

            <div className="brand-sub">
              Dynamic Railway ETA Intelligence
            </div>

          </div>

        </div>


        <div className="header-status">

          <div className="system-status">

            <span className="status-dot"></span>

            SYSTEM ONLINE

          </div>

          <div className="model-chip">

            <Brain size={14} />

            GRU + ATTENTION AI

          </div>

        </div>

      </header>


      {/* ==================================================
          HERO
      ================================================== */}

      <section className="control-section">

        <div className="control-copy">

          <div className="eyebrow">
            SMART INDIA HACKATHON 2026
          </div>

          <h1>
            Dynamic Train
            <span> ETA Intelligence</span>
          </h1>

          <p>
            Watch AI continuously update downstream
            delay predictions as the train progresses.
          </p>

        </div>


        <div className="train-controls">

          <div className="control-group">

            <label>
              <TrainFront size={14} />
              TRAIN
            </label>

            <div className="select-wrap">

              <select
                value={trainNo}
                onChange={(e) =>
                  changeTrain(
                    e.target.value
                  )
                }
              >

                {trains.map(
                  (train) => (

                    <option
                      key={train}
                      value={train}
                    >
                      {train}
                    </option>

                  )
                )}

              </select>

              <ChevronDown size={16} />

            </div>

          </div>


          <div className="control-group">

            <label>
              <Clock3 size={14} />
              JOURNEY
            </label>

            <div className="date-select">

              <select
                value={journeyDate}
                onChange={(e) =>
                  changeDate(
                    e.target.value
                  )
                }
              >

                {journeyDates.map(
                  (date) => (

                    <option
                      key={date}
                      value={date}
                    >
                      {date}
                    </option>

                  )
                )}

              </select>

            </div>

          </div>


          <button
            className={
              running
                ? "track-btn active"
                : "track-btn"
            }
            onClick={() => {

              if (running) {
                setRunning(false);
              } else {
                if (liveState && !liveState.is_finished) {
                  setRunning(true);
                } else {
                  startLive();
                }
              }

            }}
          >

            {running ? (

              <>
                <Pause size={17} />
                PAUSE
              </>

            ) : (

              <>
                <Play size={17} />
                TRACK TRAIN
              </>

            )}

          </button>

          <button
            className="track-btn reset-btn"
            title="Reset simulation to origin station"
            onClick={() => {
              startLive();
            }}
          >
            <RefreshCw size={15} />
            RESET
          </button>

        </div>

      </section>


      {/* ==================================================
          ERROR
      ================================================== */}

      {error && (

        <div className="error-banner">

          <AlertTriangle size={17} />

          <span>
            {error}
          </span>

          <button
            onClick={() =>
              setError("")
            }
          >
            ×
          </button>

        </div>

      )}


      {/* ==================================================
          TABS
      ================================================== */}

      <nav className="tabs">

        <button
          className={
            activeTab === "journey"
              ? "active"
              : ""
          }
          onClick={() =>
            setActiveTab(
              "journey"
            )
          }
        >
          <Route size={15} />
          Journey
        </button>

        <button
          className={
            activeTab === "status"
              ? "active"
              : ""
          }
          onClick={() =>
            setActiveTab(
              "status"
            )
          }
        >
          <Activity size={15} />
          Train Status
        </button>

        <button
          className={
            activeTab === "network"
              ? "active"
              : ""
          }
          onClick={() =>
            setActiveTab(
              "network"
            )
          }
        >
          <MapPin size={15} />
          Rail Network
        </button>

        <button
          className={
            activeTab === "ai"
              ? "active"
              : ""
          }
          onClick={() =>
            setActiveTab(
              "ai"
            )
          }
        >
          <Sparkles size={15} />
          AI Insights
        </button>

      </nav>


      {/* ==================================================
          STATUS CARDS
      ================================================== */}

      <section className="status-grid">

        <StatusCard
          icon={
            <MapPin size={20} />
          }
          iconClass="maroon"
          label="CURRENT STATION"
          value={
            liveState?.station_code ||
            route[0]?.station_code ||
            "—"
          }
          sub={
            liveState?.station_name ||
            route[0]?.station_name ||
            "Waiting for train"
          }
        />

        <StatusCard
          icon={
            <Clock3 size={20} />
          }
          iconClass="gold"
          label="CURRENT DELAY"
          value={
            liveState && Number.isFinite(Number(liveState.delay))
              ? `${Math.round(liveState.delay)} min`
              : "0 min"
          }
          sub="Observed / replayed"
        />

        <StatusCard
          icon={
            <Gauge size={20} />
          }
          iconClass="green"
          label="SIMULATED SPEED"
          value={
            liveState && Number.isFinite(Number(liveState.speed_kmph))
              ? `${Math.round(liveState.speed_kmph)} km/h`
              : "72 km/h"
          }
          sub="Demo telemetry"
        />

        <StatusCard
          icon={
            <TrendingUp size={20} />
          }
          iconClass="blue"
          label="JOURNEY PROGRESS"
          value={
            `${Math.round(
              progress
            )}%`
          }
          sub={
            `Station ${
              Math.min(totalStationsCount, Math.max(1, currentIndex))
            } / ${
              totalStationsCount
            }`
          }
        />

      </section>


      {/* ==================================================
          JOURNEY TAB
      ================================================== */}

      {activeTab === "journey" && (

        <>

          {/* ----------------------------------------------
              TRAIN PROGRESS
          ---------------------------------------------- */}

          <section className="panel train-progress-panel">

            <div className="panel-header">

              <div>

                <div className="eyebrow">
                  LIVE ROUTE POSITION
                </div>

                <h2>
                  Train Progress
                </h2>

                <p>
                  Click any station to inspect its
                  forecast.
                </p>

              </div>

              <div className="progress-live">

                <span className="live-dot"></span>

                {running
                  ? "MOVING"
                  : liveState
                    ? "PAUSED"
                    : "READY"}

              </div>

            </div>


            <div className="station-selector">

              <span>
                Inspect station
              </span>

              <select
                value={
                  selectedStation || ""
                }
                onChange={(e) =>
                  setSelectedStation(
                    e.target.value
                  )
                }
              >

                <option value="">
                  Select station
                </option>

                {route.map(
                  (station) => (

                    <option
                      key={
                        station.station_code
                      }
                      value={
                        station.station_code
                      }
                    >
                      {
                        station.station_code
                      } — {
                        station.station_name
                      }
                    </option>

                  )
                )}

              </select>

            </div>


            <div className="progress-route">

              <div className="route-line-bg"></div>

              <div
                className="route-line-fill"
                style={{
                  width: `${Math.min(
                    100,
                    Math.max(
                      0,
                      progress
                    )
                  )}%`,
                }}
              ></div>


              <div
                className="moving-train"
                style={{
                  left: `${Math.min(
                    100,
                    Math.max(
                      0,
                      progress
                    )
                  )}%`,
                }}
              >

                <div className="train-glow"></div>

                <div className="train-icon">

                  <TrainFront size={25} />

                </div>

              </div>


              <div className="route-stations">

                {route.map(
                  (
                    station,
                    index
                  ) => {

                    const stationIdx = Number(
                      station.journey_station_index ?? (index + 1)
                    );

                    const isCurrent =
                      stationIdx === currentIndex ||
                      station.station_code === liveState?.station_code;

                    const isSelected =
                      station.station_code ===
                      selectedStation;

                    const passed =
                      stationIdx < currentIndex;

                    return (

                      <button
                        key={`${station.station_code}-${index}`}
                        className={[
                          "route-stop",
                          isCurrent
                            ? "current"
                            : "",
                          isSelected
                            ? "selected"
                            : "",
                          passed
                            ? "passed"
                            : "",
                        ].join(" ")}
                        onClick={() =>
                          setSelectedStation(
                            station.station_code
                          )
                        }
                      >

                        <span className="station-dot"></span>

                        <span className="station-code">
                          {
                            station.station_code
                          }
                        </span>

                        <span className="station-name">
                          {
                            station.station_name
                          }
                        </span>

                      </button>

                    );

                  }
                )}

              </div>

            </div>


            {liveState && (

              <div className="current-location-banner">

                <div className="location-left">

                  <div className="location-icon">
                    <TrainFront size={19} />
                  </div>

                  <div>

                    <span>
                      TRAIN {trainNo}
                    </span>

                    <strong>
                      Currently at{" "}
                      {
                        liveState.station_name ||
                        liveState.station_code
                      }
                    </strong>

                  </div>

                </div>


                <div className="location-progress">

                  <strong>
                    {
                      Math.round(
                        progress
                      )
                    }%
                  </strong>

                  <span>
                    journey completed
                  </span>

                </div>

              </div>

            )}

          </section>


          {/* ----------------------------------------------
              SELECTED STATION
          ---------------------------------------------- */}

          {selectedStationData && (

            <section className="panel selected-station-panel">

              <div className="selected-station-icon">
                <MapPin size={20} />
              </div>

              <div className="selected-station-info">

                <span>
                  SELECTED STATION
                </span>

                <h3>
                  {
                    selectedStationData.station_name ||
                    selectedStationData.station_code
                  }
                </h3>

                <p>
                  {
                    selectedStationData.station_code
                  }
                  {" · "}
                  Station{" "}
                  {
                    selectedStationData.journey_station_index ||
                    (route.findIndex((s) => s.station_code === selectedStationData.station_code) + 1)
                  }
                </p>

              </div>


              {selectedForecast && (

                <div className="selected-delay">

                  <span>
                    AI PREDICTED DELAY
                  </span>

                  <strong>
                    {
                      Math.round(
                        selectedForecast.predicted_delay
                      )
                    } min
                  </strong>

                  <small>
                    confidence{" "}
                    {
                      Math.round(
                        Number(
                          selectedForecast.confidence ||
                          0
                        ) * 100
                      )
                    }%
                  </small>

                </div>

              )}

            </section>

          )}


          {/* ----------------------------------------------
              GRAPH
          ---------------------------------------------- */}

          <section className="panel delay-forecast-panel">

            <div className="panel-header">

              <div>

                <div className="eyebrow">
                  AI PREDICTION ENGINE
                </div>

                <h2>
                  Live Delay Forecast
                </h2>

                <p>
                  Current observation → predicted
                  downstream delay
                </p>

              </div>


              <div className="live-badge">

                <span className="live-dot"></span>

                {running
                  ? "UPDATING EVERY 3 SEC"
                  : "REPLAY READY"}

              </div>

            </div>


            <div className="chart-legend">

              <span>

                <span className="legend-dot current"></span>

                Current observation

              </span>


              <span>

                <span className="legend-dot predicted"></span>

                GRU AI forecast

              </span>

            </div>


            <div className="delay-chart">

              {graphData.length ? (

                <ResponsiveContainer
                  width="100%"
                  height={390}
                >

                  <LineChart
                    data={graphData}
                    margin={{
                      top: 25,
                      right: 25,
                      left: 10,
                      bottom: 60,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="station"
                      tick={{
                        fontSize: 11,
                      }}
                      angle={-35}
                      textAnchor="end"
                      height={80}
                    />

                    <YAxis
                      tick={{
                        fontSize: 11,
                      }}
                      label={{
                        value:
                          "Delay (minutes)",
                        angle: -90,
                        position:
                          "insideLeft",
                      }}
                    />

                    <Tooltip
                      content={({
                        active,
                        payload,
                      }) => {

                        if (
                          !active ||
                          !payload ||
                          !payload.length
                        ) {
                          return null;
                        }

                        const item =
                          payload[0].payload;

                        return (

                          <div className="chart-tooltip">

                            <strong>
                              {
                                item.stationName
                              }
                            </strong>

                            <div className="tooltip-delay">

                              {
                                Math.round(
                                  item.delay
                                )
                              }{" "}
                              minutes

                            </div>

                            <small>

                              {
                                item.type ===
                                "current"
                                  ? "Current observation"
                                  : "AI predicted delay"
                              }

                            </small>

                            {item.confidence && (

                              <small>
                                Confidence:{" "}
                                {
                                  Math.round(
                                    item.confidence *
                                    100
                                  )
                                }%
                              </small>

                            )}

                          </div>

                        );

                      }}
                    />


                    <Line
                      type="monotone"
                      dataKey="delay"
                      stroke="#a51c30"
                      strokeWidth={4}
                      dot={(props) => {

                        const {
                          cx,
                          cy,
                          payload,
                        } = props;

                        return (

                          <circle
                            cx={cx}
                            cy={cy}
                            r={
                              payload.type ===
                              "current"
                                ? 9
                                : 6
                            }
                            className={
                              payload.type ===
                              "current"
                                ? "current-chart-dot"
                                : "forecast-chart-dot"
                            }
                          />

                        );

                      }}
                      activeDot={{
                        r: 8,
                      }}
                    />

                  </LineChart>

                </ResponsiveContainer>

              ) : (

                <div className="chart-empty">

                  <TrendingUp
                    size={42}
                  />

                  <h3>
                    Start train tracking
                  </h3>

                  <p>
                    The graph will show the current
                    delay and the next four AI predictions.
                  </p>

                </div>

              )}

            </div>


            {/* ------------------------------------------
                FORECAST STATIONS
            ------------------------------------------ */}

            {forecast?.stations?.length > 0 && (

              <div className="forecast-station-strip">

                {forecast.stations.map(
                  (station) => (

                    <button
                      key={
                        station.station_code
                      }
                      className="forecast-station"
                      onClick={() =>
                        setSelectedStation(
                          station.station_code
                        )
                      }
                    >

                      <div className="forecast-step">
                        +
                        {
                          station.station_sequence -
                          (liveState?.station_sequence || 0)
                        }
                      </div>

                      <div className="forecast-station-info">

                        <strong>
                          {
                            station.station_code
                          }
                        </strong>

                        <span>
                          {
                            station.station_name
                          }
                        </span>

                      </div>

                      <div className="forecast-delay">

                        {
                          Math.round(
                            station.predicted_delay
                          )
                        }m

                      </div>

                    </button>

                  )
                )}

              </div>

            )}

          </section>


          {/* ----------------------------------------------
              AI + CONFIDENCE
          ---------------------------------------------- */}

          <div className="insight-grid">

            <section className="panel ai-panel">

              <div className="panel-header">

                <div>

                  <div className="eyebrow">
                    MODEL EXPLANATION
                  </div>

                  <h2>
                    Why did the ETA change?
                  </h2>

                </div>

                <Brain
                  size={22}
                />

              </div>


              <div className="ai-explanation">

                <div className="ai-icon">
                  <Sparkles size={22} />
                </div>

                <div>

                  <p>
                    {liveState
                      ? `The model is using the current ${
                          Number.isFinite(Number(liveState?.delay))
                            ? Math.round(Number(liveState.delay))
                            : 0
                        } minute delay together with historical train behaviour, station position and recent delay changes to estimate downstream propagation.`
                      : "Start the replay to give the forecasting engine a current train observation."
                    }
                  </p>


                  <div className="explanation-pills">

                    <span>
                      <Clock3 size={11} />
                      Current delay
                    </span>

                    <span>
                      <TrendingUp size={11} />
                      Delay history
                    </span>

                    <span>
                      <Route size={11} />
                      Station position
                    </span>

                    <span>
                      <Brain size={11} />
                      GRU prediction
                    </span>

                  </div>

                </div>

              </div>

            </section>


            <section className="panel confidence-panel">

              <div className="panel-header">

                <div>

                  <div className="eyebrow">
                    UNCERTAINTY
                  </div>

                  <h2>
                    Forecast Confidence
                  </h2>

                </div>

                <ShieldCheck
                  size={22}
                />

              </div>


              <div className="overall-confidence">

                <strong>
                  {
                    averageConfidence !== null
                      ? `${Math.round(
                          averageConfidence
                        )}%`
                      : "—"
                  }
                </strong>

                <span>
                  average confidence
                </span>

              </div>


              <div className="confidence-list">

                {forecast?.stations?.map(
                  (station) => {

                    const confidence =
                      Number(
                        station.confidence ||
                        0
                      ) * 100;

                    return (

                      <div
                        className="confidence-row"
                        key={
                          station.station_code
                        }
                      >

                        <div className="confidence-station">

                          <strong>
                            {
                              station.station_code
                            }
                          </strong>

                          <span>
                            +forecast
                          </span>

                        </div>

                        <div className="confidence-bar">

                          <div
                            className="confidence-fill"
                            style={{
                              width: `${confidence}%`,
                            }}
                          ></div>

                        </div>

                        <strong>
                          {
                            Math.round(
                              confidence
                            )
                          }%
                        </strong>

                      </div>

                    );

                  }
                )}

                {!forecast?.stations?.length && (

                  <div className="confidence-empty">
                    Forecast confidence will appear
                    after tracking starts.
                  </div>

                )}

              </div>

            </section>

          </div>


          {/* ----------------------------------------------
              QUICK ACTIONS
          ---------------------------------------------- */}

          <section className="quick-actions">

            <button
              onClick={() =>
                setAlertsOpen(true)
              }
            >
              <Bell size={18} />
              Alerts
            </button>

            <button
              onClick={() =>
                setWhatIfOpen(true)
              }
            >
              <Zap size={18} />
              What-if
            </button>

            <button
              onClick={() =>
                setAssistantOpen(true)
              }
            >
              <MessageCircle size={18} />
              AI Assistant
            </button>

            <button
              onClick={() =>
                navigator.clipboard?.writeText(
                  window.location.href
                )
              }
            >
              <Share2 size={18} />
              Share
            </button>

            <button
              onClick={() =>
                setError(
                  "Passenger ticket integration is a future module."
                )
              }
            >
              <Ticket size={18} />
              Ticket
            </button>

          </section>

        </>

      )}


      {/* ==================================================
          STATUS TAB
      ================================================== */}

      {activeTab === "status" && (

        <section className="panel status-detail-panel">

          <div className="panel-header">

            <div>

              <div className="eyebrow">
                LIVE TRAIN TELEMETRY
              </div>

              <h2>
                Train Status
              </h2>

              <p>
                Historical replay simulates the incoming
                observation stream.
              </p>

            </div>

            <Radio size={24} />

          </div>


          <div className="status-detail-grid">

            <DetailMetric
              label="TRAIN"
              value={trainNo || "—"}
            />

            <DetailMetric
              label="CURRENT STATION"
              value={
                liveState?.station_code ||
                "—"
              }
            />

            <DetailMetric
              label="DELAY"
              value={
                liveState && Number.isFinite(Number(liveState.delay))
                  ? `${Math.round(liveState.delay)} min`
                  : "0 min"
              }
            />

            <DetailMetric
              label="SPEED"
              value={
                liveState && Number.isFinite(Number(liveState.speed_kmph))
                  ? `${Math.round(liveState.speed_kmph)} km/h`
                  : "72 km/h"
              }
            />

            <DetailMetric
              label="STATION INDEX"
              value={
                liveState
                  ? `${currentIndex} / ${liveState.total_stations || 7}`
                  : "1 / 7"
              }
            />

            <DetailMetric
              label="DELAY CHANGE"
              value={
                liveState
                  ? `${
                      liveState.delay_change > 0
                        ? "+"
                        : ""
                    }${Math.round(
                      liveState.delay_change
                    )} min`
                  : "—"
              }
            />

          </div>


          <div className="telemetry-banner">

            <div className="telemetry-icon">
              <Radio size={20} />
            </div>

            <div>

              <strong>
                Historical Replay Simulation
              </strong>

              <p>
                This demo feeds historical observations
                into the same pipeline that will accept
                authorized live railway data in production.
              </p>

            </div>

          </div>

        </section>

      )}


      {/* ==================================================
          NETWORK TAB
      ================================================== */}

      {activeTab === "network" && (

        <section className="panel network-panel">

          <div className="panel-header">

            <div>

              <div className="eyebrow">
                ROUTE INTELLIGENCE
              </div>

              <h2>
                Railway Network
              </h2>

              <p>
                Complete station sequence for the selected
                journey.
              </p>

            </div>

            <Route size={24} />

          </div>


          <div className="network-grid">

            {route.map(
              (station, index) => {

                const current =
                  station.station_code ===
                  liveState?.station_code;

                const predicted =
                  forecast?.stations?.some(
                    (s) =>
                      s.station_code ===
                      station.station_code
                  );

                return (

                  <button
                    className={
                      current
                        ? "network-station current"
                        : predicted
                          ? "network-station predicted"
                          : "network-station"
                    }
                    key={`${station.station_code}-${index}`}
                    onClick={() =>
                      setSelectedStation(
                        station.station_code
                      )
                    }
                  >

                    <div className="network-number">
                      {index + 1}
                    </div>

                    <div>

                      <strong>
                        {
                          station.station_code
                        }
                      </strong>

                      <span>
                        {
                          station.station_name
                        }
                      </span>

                    </div>

                    {current && (
                      <span className="network-badge">
                        TRAIN
                      </span>
                    )}

                    {predicted && !current && (
                      <span className="network-badge">
                        AI
                      </span>
                    )}

                  </button>

                );

              }
            )}

          </div>

        </section>

      )}


      {/* ==================================================
          AI TAB
      ================================================== */}

      {activeTab === "ai" && (

        <>

          <section className="panel ai-dashboard">

            <div className="ai-dashboard-hero">

              <div className="big-ai-icon">
                <Brain size={32} />
              </div>

              <div>

                <div className="eyebrow">
                  RAILPULSE AI INTELLIGENCE
                </div>

                <h2>
                  Dynamic Delay Intelligence & GRU Attention
                </h2>

                <p>
                  The GRU + Attention model recurrently analyzes the preceding station sequence,
                  estimating downstream delay propagation while accounting for section distance,
                  arrival time velocity, and station recovery capacity.
                </p>

              </div>

            </div>


            <div className="ai-stat-grid">

              <div>
                <span>
                  MODEL ARCHITECTURE
                </span>
                <strong>
                  {forecast?.model || "GRU + Attention"}
                </strong>
              </div>

              <div>
                <span>
                  FORECAST HORIZON
                </span>
                <strong>
                  +{forecast?.stations?.length || 4} stations
                </strong>
              </div>

              <div>
                <span>
                  CURRENT DELAY
                </span>
                <strong>
                  {
                    liveState && Number.isFinite(Number(liveState.delay))
                      ? `${Math.round(liveState.delay)} min`
                      : "0 min"
                  }
                </strong>
              </div>

              <div>
                <span>
                  MAX DOWNSTREAM DELAY
                </span>
                <strong>
                  {
                    maxForecastDelay !== null
                      ? `${Math.round(maxForecastDelay)} min`
                      : `${Math.round(liveState?.delay || 0)} min`
                  }
                </strong>
              </div>

            </div>

          </section>


          {/* DOWNSTREAM PROPAGATION MATRIX TABLE */}
          <section className="panel">

            <div className="panel-header">

              <div>

                <div className="eyebrow">
                  PROPAGATION MATRIX
                </div>

                <h2>
                  Downstream Stations Delay & Risk Forecast
                </h2>

                <p>
                  Real-time station-level forecast generated by the PyTorch GRU + Attention model
                </p>

              </div>

              <Sparkles size={22} />

            </div>

            {forecast?.stations?.length ? (

              <div className="ai-table-wrap">

                <table className="ai-table">

                  <thead>

                    <tr>
                      <th>DOWNSTREAM STATION</th>
                      <th>PREDICTED DELAY</th>
                      <th>DELAY CHANGE (Δ)</th>
                      <th>95% CONFIDENCE BAND</th>
                      <th>MODEL CONFIDENCE</th>
                      <th>PROPAGATION RISK</th>
                    </tr>

                  </thead>

                  <tbody>

                    {forecast.stations.map((s) => {

                      const pred = Math.round(s.predicted_delay || 0);
                      const change = Math.round(s.predicted_change || 0);
                      const conf = Math.round((s.confidence || 0.85) * 100);
                      const lower = Math.round(s.lower_delay || 0);
                      const upper = Math.round(s.upper_delay || 0);

                      let riskClass = "green";
                      let riskText = "ON TIME / RECOVERED";
                      if (pred > 30) {
                        riskClass = "red";
                        riskText = "SEVERE DELAY";
                      } else if (pred > 15) {
                        riskClass = "orange";
                        riskText = "MODERATE RISK";
                      } else if (pred > 5) {
                        riskClass = "blue";
                        riskText = "LOW DELAY IMPACT";
                      }

                      return (

                        <tr key={s.station_code}>

                          <td>
                            <span className="station-badge">
                              {s.station_code}
                            </span>
                            {" "}
                            <strong>
                              {s.station_name || s.station_code}
                            </strong>
                          </td>

                          <td>
                            <span className={`predicted-delay-badge ${riskClass}`}>
                              {pred} min
                            </span>
                          </td>

                          <td>
                            <strong style={{ color: change > 0 ? "#b91c1c" : change < 0 ? "#16733d" : "#555" }}>
                              {change > 0 ? `+${change}` : change} min
                            </strong>
                          </td>

                          <td>
                            {lower}m to {upper}m
                          </td>

                          <td>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <div className="attention-bar-bg" style={{ width: "70px" }}>
                                <div
                                  className="attention-bar-fill"
                                  style={{ width: `${conf}%` }}
                                ></div>
                              </div>
                              <strong>{conf}%</strong>
                            </div>
                          </td>

                          <td>
                            <span className={`risk-pill ${riskClass}`}>
                              {riskText}
                            </span>
                          </td>

                        </tr>

                      );

                    })}

                  </tbody>

                </table>

              </div>

            ) : (

              <div style={{ padding: "30px", textAlign: "center", color: "#888" }}>
                Start train tracking to generate real-time station forecast matrix.
              </div>

            )}

          </section>


          {/* NEURAL NETWORK ARCHITECTURE & ATTENTION FACTORS */}
          <section className="panel">

            <div className="panel-header">

              <div>

                <div className="eyebrow">
                  NEURAL SPECIFICATIONS
                </div>

                <h2>
                  GRU Architecture & Attention Weights
                </h2>

                <p>
                  Internal network mechanics and sequence weight attribution
                </p>

              </div>

              <Brain size={22} />

            </div>

            <div className="arch-grid">

              <div className="arch-card">
                <span>RECURRENT CORE</span>
                <strong>2-Layer GRU</strong>
                <p>Hidden dimension 128 with dropout (0.25) to preserve sequential state memory.</p>
              </div>

              <div className="arch-card">
                <span>ATTENTION LAYER</span>
                <strong>Multi-Head Temporal</strong>
                <p>Softmax-weighted dynamic attention focusing on pivotal station transitions.</p>
              </div>

              <div className="arch-card">
                <span>SEQUENCE WINDOW</span>
                <strong>6 Stations Lookback</strong>
                <p>Captures historical propagation momentum across recent section traversals.</p>
              </div>

              <div className="arch-card">
                <span>STABILITY DAMPING</span>
                <strong>1 / √step Decay</strong>
                <p>Prevents runaway variance accumulation across multi-horizon downstream forecasts.</p>
              </div>

            </div>

            <div className="attention-grid">

              <div className="attention-item">
                <div className="attention-top">
                  <span>Observed Delay Velocity (Recent Station Δ)</span>
                  <span>94%</span>
                </div>
                <div className="attention-bar-bg">
                  <div className="attention-bar-fill" style={{ width: "94%" }}></div>
                </div>
              </div>

              <div className="attention-item">
                <div className="attention-top">
                  <span>Current Station Arrival Anchor</span>
                  <span>89%</span>
                </div>
                <div className="attention-bar-bg">
                  <div className="attention-bar-fill" style={{ width: "89%" }}></div>
                </div>
              </div>

              <div className="attention-item">
                <div className="attention-top">
                  <span>Section Distance & Inter-Station Clearance</span>
                  <span>78%</span>
                </div>
                <div className="attention-bar-bg">
                  <div className="attention-bar-fill" style={{ width: "78%" }}></div>
                </div>
              </div>

              <div className="attention-item">
                <div className="attention-top">
                  <span>Temporal & Day-of-Week Disruption Patterns</span>
                  <span>65%</span>
                </div>
                <div className="attention-bar-bg">
                  <div className="attention-bar-fill" style={{ width: "65%" }}></div>
                </div>
              </div>

            </div>

          </section>


          {/* FEATURE EXPLAINABILITY */}
          <section className="panel">

            <div className="panel-header">

              <div>

                <div className="eyebrow">
                  EXPLAINABILITY
                </div>

                <h2>
                  What influences the prediction?
                </h2>

              </div>

              <Sparkles size={22} />

            </div>


            <div className="feature-grid">

              <Feature
                icon={
                  <Clock3 size={20} />
                }
                title="Current Delay"
                text="The latest observed train delay anchors the downstream prediction."
              />

              <Feature
                icon={
                  <TrendingUp size={20} />
                }
                title="Delay Change"
                text="Recent delay increases or recovery indicate whether disruption is propagating."
              />

              <Feature
                icon={
                  <Route size={20} />
                }
                title="Station Position"
                text="The model learns different behaviour across the train's journey."
              />

              <Feature
                icon={
                  <Brain size={20} />
                }
                title="Historical Patterns"
                text="Train and station-specific historical behaviour provides context."
              />

            </div>

          </section>

        </>

      )}


      {/* ==================================================
          FOOTER
      ================================================== */}

      <footer>

        <div>

          <strong>
            RailPulse AI
          </strong>

          SIH26028 · Dynamic ETA Forecasting

        </div>

        <div className="footer-right">

          <span>
            HISTORICAL REPLAY
          </span>

          <span>
            •
          </span>

          <span>
            GRU + ATTENTION AI
          </span>

          <span>
            •
          </span>

          <span>
            API READY
          </span>

        </div>

      </footer>


      {/* ==================================================
          AI ASSISTANT MODAL
      ================================================== */}

      {assistantOpen && (

        <Modal
          title="RailPulse AI Assistant"
          icon={
            <MessageCircle size={20} />
          }
          onClose={() =>
            setAssistantOpen(false)
          }
        >

          <div className="assistant-chat">

            <div className="assistant-message">

              <Brain size={18} />

              <span>
                Ask me about the train's delay,
                next stations, or why the forecast changed.
              </span>

            </div>

            {assistantReply && (

              <div className="assistant-reply">

                <Sparkles size={18} />

                <span>
                  {assistantReply}
                </span>

              </div>

            )}

          </div>


          <div className="assistant-input">

            <input
              value={assistantMessage}
              onChange={(e) =>
                setAssistantMessage(
                  e.target.value
                )
              }
              onKeyDown={(e) => {

                if (
                  e.key ===
                  "Enter"
                ) {
                  askAssistant();
                }

              }}
              placeholder="e.g. Why is the train delayed?"
            />

            <button
              onClick={
                askAssistant
              }
            >
              <Send size={17} />
            </button>

          </div>

        </Modal>

      )}


      {/* ==================================================
          WHAT IF MODAL
      ================================================== */}

      {whatIfOpen && (

        <Modal
          title="What-if Simulator"
          icon={
            <Zap size={20} />
          }
          onClose={() =>
            setWhatIfOpen(false)
          }
        >

          <p className="modal-description">

            Simulate an additional disruption and see
            the immediate delay impact.

          </p>


          <div className="what-if-control">

            <label>
              Additional disruption
            </label>

            <div className="range-value">
              +{whatIfDelay} minutes
            </div>

            <input
              type="range"
              min="0"
              max="120"
              step="5"
              value={whatIfDelay}
              onChange={(e) =>
                setWhatIfDelay(
                  Number(
                    e.target.value
                  )
                )
              }
            />

          </div>


          <div className="what-if-result">

            <span>
              Simulated current delay
            </span>

            <strong>
              {whatIfResult} min
            </strong>

            <small>
              This is a scenario calculation, not
              a replacement for the trained forecast.
            </small>

          </div>

        </Modal>

      )}


      {/* ==================================================
          ALERTS MODAL
      ================================================== */}

      {alertsOpen && (

        <Modal
          title="Journey Alerts"
          icon={
            <Bell size={20} />
          }
          onClose={() =>
            setAlertsOpen(false)
          }
        >

          <div className="alert-item">

            <div className="alert-icon">
              <AlertTriangle size={18} />
            </div>

            <div>

              <strong>
                Delay monitoring active
              </strong>

              <p>
                RailPulse is continuously checking the
                current observation against downstream
                predictions.
              </p>

            </div>

          </div>


          {delayTrend > 10 && (

            <div className="alert-item warning">

              <div className="alert-icon">
                <TrendingUp size={18} />
              </div>

              <div>

                <strong>
                  Delay propagation detected
                </strong>

                <p>
                  The forecast indicates increasing
                  downstream delay.
                </p>

              </div>

            </div>

          )}


          {delayTrend <= 10 && (

            <div className="alert-item good">

              <div className="alert-icon">
                <ShieldCheck size={18} />
              </div>

              <div>

                <strong>
                  No major propagation signal
                </strong>

                <p>
                  Current forecast does not show a strong
                  increase beyond the current delay.
                </p>

              </div>

            </div>

          )}

        </Modal>

      )}

    </div>

  );

}


// ============================================================
// COMPONENTS
// ============================================================

function StatusCard({
  icon,
  iconClass,
  label,
  value,
  sub,
}) {

  return (

    <div className="status-card">

      <div
        className={`status-icon ${iconClass}`}
      >
        {icon}
      </div>

      <div>

        <div className="metric-label">
          {label}
        </div>

        <div className="metric-value">
          {value}
        </div>

        <div className="metric-sub">
          {sub}
        </div>

      </div>

    </div>

  );

}


function DetailMetric({
  label,
  value,
}) {

  return (

    <div className="detail-metric">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>

  );

}


function Feature({
  icon,
  title,
  text,
}) {

  return (

    <div className="feature-card">

      <div className="feature-icon">
        {icon}
      </div>

      <div>

        <strong>
          {title}
        </strong>

        <p>
          {text}
        </p>

      </div>

    </div>

  );

}


function Modal({
  title,
  icon,
  onClose,
  children,
}) {

  return (

    <div
      className="modal-backdrop"
      onClick={onClose}
    >

      <div
        className="modal"
        onClick={(e) =>
          e.stopPropagation()
        }
      >

        <div className="modal-header">

          <div>

            <div className="modal-icon">
              {icon}
            </div>

            <h2>
              {title}
            </h2>

          </div>

          <button
            className="modal-close"
            onClick={onClose}
          >
            ×
          </button>

        </div>

        <div className="modal-body">
          {children}
        </div>

      </div>

    </div>

  );

}


export default App;