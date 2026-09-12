import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Gauge,
  MapPin,
  Radio,
  ShieldCheck,
  TrainFront,
  TrendingUp,
  Clock3,
  Zap,
} from "lucide-react";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

import "./App.css";


const TRAIN_DATA = {
  "12314": {
    name: "Sealdah Rajdhani",
    current: "NDLS",
    delay: 54,
    propagation: 0.979,
    stations: [
      "NDLS",
      "CNB",
      "DDU",
      "GAYA",
      "DHN",
      "ASN",
      "DGR",
      "SDAH",
    ],
  },

  "12428": {
    name: "Rewa Express",
    current: "ANVT",
    delay: 51,
    propagation: 0.883,
    stations: [
      "ANVT",
      "REWA",
      "STA",
      "JTW",
      "DBR",
      "SRJ",
      "PRYJ",
      "MKP",
      "FTP",
      "CNB",
      "JJK",
      "ALJN",
      "GZB",
      "SRO",
    ],
  },

  "12628": {
    name: "Karnataka Express",
    current: "NDLS",
    delay: 65,
    propagation: 0.956,
    stations: [
      "NDLS",
      "MTJ",
      "AGC",
      "GWL",
      "VGLJ",
      "BINA",
      "BPL",
      "ET",
      "KNW",
      "BAU",
      "BSL",
      "JL",
      "MMR",
      "KPG",
      "BAP",
      "ANG",
      "DD",
      "KWV",
      "SUR",
      "KLBG",
      "WADI",
      "YG",
      "RC",
      "MALM",
      "AD",
      "GTL",
      "ATP",
      "DMM",
      "SSPN",
      "PKD",
      "HUP",
      "YNK",
      "BNC",
      "SBC",
    ],
  },

  "12802": {
    name: "Puri Express",
    current: "PURI",
    delay: 79,
    propagation: 0.930,
    stations: [
      "PURI",
      "BBS",
      "KUR",
      "CTC",
      "HIJ",
    ],
  },

  "12952": {
    name: "Mumbai Rajdhani",
    current: "NDLS",
    delay: 16,
    propagation: 0.908,
    stations: [
      "NDLS",
      "KOTA",
      "NAD",
      "RTM",
      "BRC",
      "ST",
      "BVI",
      "MMCT",
    ],
  },

  "12958": {
    name: "Swarna Jayanti Rajdhani",
    current: "NDLS",
    delay: 7,
    propagation: 0.773,
    stations: [
      "NDLS",
      "KOTA",
      "RTM",
      "BRC",
      "ST",
      "BVI",
      "MMCT",
    ],
  },

  "14310": {
    name: "Ujjain Express",
    current: "START",
    delay: 50,
    propagation: 0.932,
    stations: [
      "START",
      "STN02",
      "STN03",
      "STN04",
      "STN05",
    ],
  },

  "20807": {
    name: "Vande Bharat / Express",
    current: "START",
    delay: 112,
    propagation: 0.889,
    stations: [
      "START",
      "STN02",
      "STN03",
      "STN04",
      "STN05",
      "END",
    ],
  },
};


function generateChart(delay) {
  return [
    { station: "−4", delay: Math.max(0, delay - 25) },
    { station: "−3", delay: Math.max(0, delay - 17) },
    { station: "−2", delay: Math.max(0, delay - 8) },
    { station: "Current", delay },
    { station: "+1", delay: delay + 5 },
    { station: "+2", delay: delay + 8 },
    { station: "+3", delay: delay + 12 },
    { station: "+4", delay: delay + 16 },
  ];
}


function App() {
  const [selectedTrain, setSelectedTrain] = useState("12952");

  const train = TRAIN_DATA[selectedTrain];

  const chartData = useMemo(
    () => generateChart(train.delay),
    [train.delay]
  );

  const predictedDelay = Math.round(
    train.delay * train.propagation
  );

  const confidence = Math.round(
    70 + train.propagation * 25
  );

  return (
    <div className="app">

      {/* Background grid */}
      <div className="grid-bg" />

      {/* Scanline */}
      <div className="scanline" />

      {/* ================================================= */}
      {/* NAVBAR */}
      {/* ================================================= */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-icon">
            <TrainFront size={22} />
          </div>

          <div>
            <div className="brand-name">
              RAIL<span>NET</span>
            </div>

            <div className="brand-sub">
              ETA INTELLIGENCE SYSTEM
            </div>
          </div>

        </div>


        <div className="system-status">

          <div className="status-dot" />

          <span>SYSTEM ONLINE</span>

          <span className="separator">|</span>

          <span>DATA ENGINE v0.1</span>

        </div>

      </nav>


      {/* ================================================= */}
      {/* HERO */}
      {/* ================================================= */}

      <main>

        <section className="hero">

          <div>

            <div className="eyebrow">
              <Radio size={15} />
              DYNAMIC RAILWAY INTELLIGENCE
            </div>

            <h1>
              Predict the delay.
              <br />

              <span>Before the train arrives.</span>
            </h1>

            <p>
              A dynamic ETA intelligence engine that learns how
              delays propagate across railway journeys.
            </p>

          </div>


          <div className="hero-terminal">

            <div className="terminal-header">

              <span>ENGINE STATUS</span>

              <span className="live">
                ● LIVE
              </span>

            </div>

            <div className="terminal-body">

              <div>
                <span>MODEL</span>
                <strong>ETA-ENGINE</strong>
              </div>

              <div>
                <span>MODE</span>
                <strong>REPLAY</strong>
              </div>

              <div>
                <span>LATENCY</span>
                <strong>12ms</strong>
              </div>

            </div>

          </div>

        </section>


        {/* ================================================= */}
        {/* KPI CARDS */}
        {/* ================================================= */}

        <section className="kpis">

          <KPI
            icon={<TrainFront />}
            label="TRAINS"
            value="8"
            detail="Active datasets"
          />

          <KPI
            icon={<Activity />}
            label="JOURNEYS"
            value="2,451"
            detail="Historical journeys"
          />

          <KPI
            icon={<Gauge />}
            label="OBSERVATIONS"
            value="50,598"
            detail="Station events"
          />

          <KPI
            icon={<TrendingUp />}
            label="PROPAGATION"
            value={`${(train.propagation * 100).toFixed(1)}%`}
            detail="Current train correlation"
          />

        </section>


        {/* ================================================= */}
        {/* TRAIN CONTROL */}
        {/* ================================================= */}

        <section className="control-panel">

          <div className="panel-header">

            <div>
              <span className="panel-kicker">
                TRAIN CONTROL
              </span>

              <h2>
                Journey intelligence
              </h2>
            </div>

            <select
              value={selectedTrain}
              onChange={(e) =>
                setSelectedTrain(e.target.value)
              }
            >

              {Object.keys(TRAIN_DATA).map((id) => (
                <option key={id} value={id}>
                  Train {id}
                </option>
              ))}

            </select>

          </div>


          {/* TRAIN INFO */}

          <div className="train-info">

            <div className="train-number">
              <span>TRAIN</span>
              <strong>{selectedTrain}</strong>
            </div>

            <div>
              <span>SERVICE</span>
              <strong>{train.name}</strong>
            </div>

            <div>
              <span>CURRENT DELAY</span>
              <strong className="delay">
                +{train.delay} min
              </strong>
            </div>

            <div>
              <span>PREDICTED DOWNSTREAM</span>
              <strong>
                +{predictedDelay} min
              </strong>
            </div>

            <div>
              <span>CONFIDENCE</span>
              <strong>
                {confidence}%
              </strong>
            </div>

          </div>


          {/* ================================================= */}
          {/* ROUTE */}
          {/* ================================================= */}

          <div className="route-section">

            <div className="route-title">

              <span>
                <MapPin size={15} />
                LIVE JOURNEY TRACE
              </span>

              <span>
                {train.stations.length} STATIONS
              </span>

            </div>


            <div className="route">

              <div className="route-line" />

              {train.stations.map(
                (station, index) => {

                  const current =
                    index === 0;

                  return (

                    <motion.div
                      key={station}
                      className={
                        current
                          ? "station current"
                          : "station"
                      }
                      initial={{
                        opacity: 0,
                        scale: 0.5
                      }}
                      animate={{
                        opacity: 1,
                        scale: 1
                      }}
                      transition={{
                        delay: index * 0.06
                      }}
                    >

                      <div className="station-dot">

                        {current && (
                          <motion.div
                            className="pulse"
                            animate={{
                              scale: [1, 1.8, 1],
                              opacity: [1, 0, 1]
                            }}
                            transition={{
                              duration: 1.8,
                              repeat: Infinity
                            }}
                          />
                        )}

                      </div>

                      <span>
                        {station}
                      </span>

                    </motion.div>

                  );

                }
              )}

            </div>

          </div>

        </section>


        {/* ================================================= */}
        {/* ANALYTICS GRID */}
        {/* ================================================= */}

        <section className="analytics-grid">

          {/* DELAY PROPAGATION */}

          <div className="panel chart-panel">

            <div className="panel-header">

              <div>

                <span className="panel-kicker">
                  PROPAGATION MODEL
                </span>

                <h2>
                  Delay trajectory
                </h2>

              </div>

              <Zap size={20} />

            </div>


            <div className="chart">

              <ResponsiveContainer
                width="100%"
                height={300}
              >

                <AreaChart data={chartData}>

                  <defs>

                    <linearGradient
                      id="delayGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >

                      <stop
                        offset="0%"
                        stopOpacity={0.35}
                      />

                      <stop
                        offset="100%"
                        stopOpacity={0}
                      />

                    </linearGradient>

                  </defs>

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.08)"
                  />

                  <XAxis
                    dataKey="station"
                    stroke="#7d8794"
                  />

                  <YAxis
                    stroke="#7d8794"
                  />

                  <Tooltip />

                  <Area
                    type="monotone"
                    dataKey="delay"
                    stroke="#f5b800"
                    fill="url(#delayGradient)"
                    strokeWidth={2}
                  />

                </AreaChart>

              </ResponsiveContainer>

            </div>

          </div>


          {/* ETA CARD */}

          <div className="panel eta-panel">

            <span className="panel-kicker">
              PREDICTIVE ETA
            </span>

            <h2>
              Downstream forecast
            </h2>

            <div className="eta-number">

              +{predictedDelay}

              <span>
                min
              </span>

            </div>

            <p>
              Estimated propagated delay
              after the current observation.
            </p>


            <div className="confidence">

              <div className="confidence-header">

                <span>
                  MODEL CONFIDENCE
                </span>

                <strong>
                  {confidence}%
                </strong>

              </div>

              <div className="confidence-bar">

                <motion.div
                  initial={{ width: 0 }}
                  animate={{
                    width: `${confidence}%`
                  }}
                  transition={{
                    duration: 1
                  }}
                />

              </div>

            </div>


            <div className="eta-status">

              <ShieldCheck size={18} />

              Forecast within
              expected propagation range

            </div>

          </div>

        </section>


        {/* ================================================= */}
        {/* JOURNEY TIMELINE */}
        {/* ================================================= */}

        <section className="panel timeline-panel">

          <div className="panel-header">

            <div>

              <span className="panel-kicker">
                HISTORICAL REPLAY ENGINE
              </span>

              <h2>
                Journey timeline
              </h2>

            </div>

            <Clock3 size={20} />

          </div>


          <div className="timeline">

            {[12, 18, 24, 31, 37, 44].map(
              (delay, index) => (

                <motion.div
                  className="timeline-event"
                  key={index}
                  initial={{
                    opacity: 0,
                    y: 20
                  }}
                  whileInView={{
                    opacity: 1,
                    y: 0
                  }}
                  transition={{
                    delay: index * 0.1
                  }}
                >

                  <div className="timeline-dot" />

                  <span>
                    STN {String(index + 1).padStart(2, "0")}
                  </span>

                  <strong>
                    +{delay}m
                  </strong>

                </motion.div>

              )
            )}

          </div>

        </section>


        {/* ================================================= */}
        {/* FOOTER */}
        {/* ================================================= */}

        <footer>

          <div>
            RAILNET / SIH26028
          </div>

          <div>
            DYNAMIC FORECAST OF EXPECTED TIME OF ARRIVAL
          </div>

          <div>
            <span className="status-dot" />
            ENGINE READY
          </div>

        </footer>

      </main>

    </div>
  );
}


function KPI({
  icon,
  label,
  value,
  detail
}) {

  return (

    <motion.div
      className="kpi"
      whileHover={{
        y: -5
      }}
    >

      <div className="kpi-icon">
        {icon}
      </div>

      <div>

        <span>
          {label}
        </span>

        <strong>
          {value}
        </strong>

        <small>
          {detail}
        </small>

      </div>

    </motion.div>

  );
}


export default App;