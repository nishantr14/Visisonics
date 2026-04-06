import codecs

with codecs.open('c:\\Visionics OT\\otshield\\frontend\\index.html', 'r', 'utf-8') as f:
    text = f.read()

# 1. Hero text
text = text.replace(
    "The only industrial security system that watches the network, listens to the machines, and\\n        sees with cameras — simultaneously. Coordinated attacks can't hide when all three streams are watching.",
    "The only industrial security system that correlates SCADA network traffic with physical sensor telemetry — simultaneously. Coordinated attacks can't hide when both critical layers are monitored."
)

# 2. Ticker
ticker_old = """      <div class="ticker-item">Stream A — SCADA network monitoring</div>
      <div class="ticker-item">Stream B — Acoustic anomaly detection</div>
      <div class="ticker-item">Stream C — Visual threat recognition</div>
      <div class="ticker-item">Fusion engine — Real-time correlation</div>
      <div class="ticker-item">SHAP explainability — Operator clarity</div>
      <div class="ticker-item">BATADAL + TON_IoT — Labeled cyberattack scenarios</div>
      <div class="ticker-item">MIMII Dataset — Hitachi industrial audio</div>"""
ticker_new = """      <div class="ticker-item">Stream A — SCADA network monitoring</div>
      <div class="ticker-item">Stream B — Physical sensor telemetry</div>
      <div class="ticker-item">Hardware-in-the-loop — OpenPLC Modbus</div>
      <div class="ticker-item">Fusion engine — Real-time correlation</div>
      <div class="ticker-item">SHAP explainability — Operator clarity</div>
      <div class="ticker-item">BATADAL + UNSW-NB15 — Labeled threat vectors</div>"""
text = text.replace(ticker_old, ticker_new)

# 3. Problem Section
text = text.replace(
    "fuse network + acoustic + visual into a\\n              single correlated risk score",
    "fuse network traffic + physical sensor telemetry into a\\n              single correlated risk score"
)

# 4. Pipeline Section
text = text.replace("Three streams.", "Dual streams.")
text = text.replace("three parallel AI\\n      pipelines that converge", "two parallel AI\\n      pipelines that converge")
text = text.replace("grid-template-columns: 1fr 1fr 1fr;", "grid-template-columns: 1fr 1fr;")
text = text.replace("grid-template-columns: 1fr 1fr 1fr 180px 120px;", "grid-template-columns: 1fr 1fr 180px 120px;")


pipeline_old = """    <div class="pipeline">
      <div class="pipe-card cyber fade-up">
        <div class="pipe-num">Stream A // Cyber</div>
        <div class="pipe-title">Network<br>Intelligence</div>
        <div class="pipe-subtitle">SCADA traffic analysis</div>
        <p class="pipe-desc">Captures all ICS network packets and sensor telemetry in real time. Isolation Forest and
          Autoencoder models learn what legitimate operations look like and flag deviations — command injection, replay
          attacks, and sensor manipulation.</p>
        <span class="pipe-tag">Isolation Forest</span>
        <span class="pipe-tag">BATADAL + TON_IoT</span>
        <span class="pipe-tag">Modbus / DNP3</span>
      </div>
      <div class="pipe-card audio fade-up delay-1">
        <div class="pipe-num">Stream B // Physical</div>
        <div class="pipe-title">Acoustic<br>Intelligence</div>
        <div class="pipe-subtitle">Machine sound monitoring</div>
        <p class="pipe-desc">Every industrial machine has a healthy acoustic signature. An Autoencoder trained on
          Hitachi's own MIMII dataset detects when pump, fan, valve or rail sounds deviate — catching physical tampering
          and equipment failure simultaneously.</p>
        <span class="pipe-tag">Autoencoder</span>
        <span class="pipe-tag">MIMII by Hitachi</span>
        <span class="pipe-tag">Mel Spectrogram</span>
      </div>
      <div class="pipe-card visual fade-up delay-2">
        <div class="pipe-num">Stream C // Visual</div>
        <div class="pipe-title">Visual<br>Intelligence</div>
        <div class="pipe-subtitle">Camera feed analysis</div>
        <p class="pipe-desc">Taps into existing industrial CCTV infrastructure — no new hardware required. YOLO detects
          sparks, smoke, misaligned equipment and abnormal machine states in real time across every camera feed
          simultaneously.</p>
        <span class="pipe-tag">YOLOv8</span>
        <span class="pipe-tag">Existing CCTV</span>
        <span class="pipe-tag">Real-time CV</span>
      </div>
    </div>"""
pipeline_new = """    <div class="pipeline">
      <div class="pipe-card cyber fade-up">
        <div class="pipe-num">Stream A // Cyber</div>
        <div class="pipe-title">Network<br>Intelligence</div>
        <div class="pipe-subtitle">SCADA traffic analysis</div>
        <p class="pipe-desc">Captures all ICS network packets in real time. Deep Learning Models (Isolation Forests) learn what legitimate Modbus and TCP traffic looks like and flag deviations — command injection, replay attacks, and unauthorized probes.</p>
        <span class="pipe-tag">Isolation Forest</span>
        <span class="pipe-tag">UNSW-NB15 / TON_IoT</span>
        <span class="pipe-tag">Modbus TCP</span>
      </div>
      <div class="pipe-card audio fade-up delay-1" style="border-color: #00CC88;">
        <div class="pipe-num" style="color:#00CC88;">Stream B // Physical</div>
        <div class="pipe-title">Physical<br>Intelligence</div>
        <div class="pipe-subtitle">Sensor telemetry monitoring</div>
        <p class="pipe-desc">Monitors the actual physical constraints of the water treatment plant. Analyzes real-time fluid dynamics (tank levels, pressure, flow rate) using Machine Learning to catch when a physical system is behaving impossibly or dangerously.</p>
        <span class="pipe-tag">Isolation Forest</span>
        <span class="pipe-tag">BATADAL Dataset</span>
        <span class="pipe-tag">PLC Telemetry</span>
      </div>
    </div>"""
text = text.replace(pipeline_old, pipeline_new)

# 5. Fusion Section
text = text.replace("When two streams<br>agree", "When both streams<br>agree")
text = text.replace("correlates all three", "correlates both streams")
text = text.replace("all three elevate", "both streams elevate")

matrix_old = """        <div class="matrix-header">
          <div>Cyber stream</div>
          <div>Audio stream</div>
          <div>Visual stream</div>
          <div>Classification</div>
          <div>Alert level</div>
        </div>
        <div class="matrix-row">
          <div class="level-none">Low</div>
          <div class="level-none">Low</div>
          <div class="level-none">Low</div>
          <div>Normal operations</div>
          <div class="level-none">None</div>
        </div>
        <div class="matrix-row">
          <div class="level-yellow">High</div>
          <div class="level-none">Low</div>
          <div class="level-none">Low</div>
          <div>Network probe — early stage</div>
          <div class="level-yellow">Orange</div>
        </div>
        <div class="matrix-row">
          <div class="level-none">Low</div>
          <div class="level-yellow">High</div>
          <div class="level-none">Low</div>
          <div>Equipment degradation</div>
          <div class="level-yellow">Yellow</div>
        </div>
        <div class="matrix-row">
          <div class="level-orange">High</div>
          <div class="level-orange">High</div>
          <div class="level-none">Low</div>
          <div>Coordinated cyber-physical attack</div>
          <div class="level-red">Red</div>
        </div>
        <div class="matrix-row">
          <div class="level-red">High</div>
          <div class="level-red">High</div>
          <div class="level-red">High</div>
          <div>Full attack — Ukraine pattern</div>
          <div class="level-red">Critical</div>
        </div>"""
matrix_new = """        <div class="matrix-header">
          <div>Cyber stream</div>
          <div>Physical stream</div>
          <div>Classification</div>
          <div>Alert level</div>
        </div>
        <div class="matrix-row">
          <div class="level-none">Low</div>
          <div class="level-none">Low</div>
          <div>Normal operations</div>
          <div class="level-none">None</div>
        </div>
        <div class="matrix-row">
          <div class="level-yellow">High</div>
          <div class="level-none">Low</div>
          <div>Network probe — early stage</div>
          <div class="level-yellow">Orange</div>
        </div>
        <div class="matrix-row">
          <div class="level-none">Low</div>
          <div class="level-yellow">High</div>
          <div>Sensor degradation / Drift</div>
          <div class="level-yellow">Yellow</div>
        </div>
        <div class="matrix-row">
          <div class="level-orange">High</div>
          <div class="level-orange">High</div>
          <div>Coordinated cyber-physical attack</div>
          <div class="level-red">Red</div>
        </div>
        <div class="matrix-row">
          <div class="level-red">High</div>
          <div class="level-red">High</div>
          <div>Full attack — Ukraine pattern</div>
          <div class="level-red">Critical</div>
        </div>"""
text = text.replace(matrix_old, matrix_new)

# 6. SHAP Section
shap_old = """        <div class="shap-row">
          <div class="shap-label">→ Audio: Pump-2 acoustic deviation — 73% abnormal</div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill" style="width:38%;background:#00CC88"></div>
          </div>
          <div class="shap-pct">38%</div>
        </div>
        <div class="shap-row">
          <div class="shap-label">→ Visual: Thermal anomaly detected — Camera 4</div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill" style="width:15%;background:#9966FF"></div>
          </div>
          <div class="shap-pct">15%</div>
        </div>"""
shap_new = """        <div class="shap-row" style="margin-bottom:0">
          <div class="shap-label">→ Physical: Tank-1 outflow pressure — 73% above normal</div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill" style="width:53%;background:#00CC88"></div>
          </div>
          <div class="shap-pct">53%</div>
        </div>"""
text = text.replace(shap_old, shap_new)

# 7. Tech Stack Section
text = text.replace("grid-template-columns: repeat(4, 1fr);", "grid-template-columns: repeat(3, 1fr);")

tech_old = """    <div class="tech-grid">
      <div class="tech-item fade-up">
        <div class="tech-role">Detection Model</div>
        <div class="tech-name">Ensemble RF + GBM</div>
        <div class="tech-desc">Supervised ensemble on 43 BATADAL sensors + 71 engineered features. CV F1 = 0.795, AUC =
          0.967.</div>
      </div>
      <div class="tech-item fade-up delay-1">
        <div class="tech-role">Audio model</div>
        <div class="tech-name">Autoencoder</div>
        <div class="tech-desc">Reconstruction error on Hitachi MIMII machine audio signals.</div>
      </div>
      <div class="tech-item fade-up delay-2">
        <div class="tech-role">Visual model</div>
        <div class="tech-name">YOLOv8</div>
        <div class="tech-desc">Real-time industrial anomaly detection on existing CCTV feeds.</div>
      </div>
      <div class="tech-item fade-up delay-3">
        <div class="tech-role">Explainability</div>
        <div class="tech-name">SHAP</div>
        <div class="tech-desc">Every alert explained. Which stream flagged it, why, and what to do.</div>
      </div>
      <div class="tech-item fade-up">
        <div class="tech-role">Backend</div>
        <div class="tech-name">FastAPI</div>
        <div class="tech-desc">WebSocket streaming of real-time risk scores to the dashboard.</div>
      </div>
      <div class="tech-item fade-up delay-1">
        <div class="tech-role">Frontend</div>
        <div class="tech-name">Vanilla JS + Chart.js</div>
        <div class="tech-desc">Live threat dashboard with real-time WebSocket streaming, risk gauge, and alert log.
        </div>
      </div>
      <div class="tech-item fade-up delay-2">
        <div class="tech-role">Cyber dataset</div>
        <div class="tech-name">BATADAL + TON_IoT</div>
        <div class="tech-desc">Water treatment plant sensor data + ICS network traffic logs with labeled attack windows.
        </div>
      </div>
      <div class="tech-item fade-up delay-3">
        <div class="tech-role">Audio dataset</div>
        <div class="tech-name">MIMII</div>
        <div class="tech-desc">Hitachi's own industrial machine sound dataset. Pumps, fans, valves, rails.</div>
      </div>
    </div>"""
tech_new = """    <div class="tech-grid">
      <div class="tech-item fade-up">
        <div class="tech-role">Physical Model</div>
        <div class="tech-name">Isolation Forest</div>
        <div class="tech-desc">Unsupervised anomaly detection on BATADAL engineered features.</div>
      </div>
      <div class="tech-item fade-up delay-1">
        <div class="tech-role">Cyber Model</div>
        <div class="tech-name">Isolation Forest</div>
        <div class="tech-desc">Real-time network packet and payload inspection for malicious behavior. F1 = 0.99.</div>
      </div>
      <div class="tech-item fade-up delay-2">
        <div class="tech-role">Explainability</div>
        <div class="tech-name">SHAP</div>
        <div class="tech-desc">Every alert explained. Which layer flagged it (Network vs Sensors), and exactly how.</div>
      </div>
      <div class="tech-item fade-up">
        <div class="tech-role">Backend + Simulator</div>
        <div class="tech-name">FastAPI + PyModbus</div>
        <div class="tech-desc">Hardware-In-The-Loop emulation. We actively spoof Modbus TCP packets to an OpenPLC backend.</div>
      </div>
      <div class="tech-item fade-up delay-1">
        <div class="tech-role">Frontend</div>
        <div class="tech-name">Vanilla JS + Chart.js</div>
        <div class="tech-desc">Live threat dashboard with real-time WebSocket streaming, pulsing alerts, and logs.</div>
      </div>
      <div class="tech-item fade-up delay-2">
        <div class="tech-role">Datasets</div>
        <div class="tech-name">UNSW-NB15 + BATADAL</div>
        <div class="tech-desc">Australian cyber network flows + SCADA water treatment plant telemetry.</div>
      </div>
    </div>"""
text = text.replace(tech_old, tech_new)

# 8. Datasets Section
dataset_old = """        <div class="dataset-item">
          <div class="dataset-name">BATADAL + TON_IoT</div>
          <div class="dataset-desc">BATADAL: Water treatment plant sensor telemetry with labeled attack periods.
            TON_IoT: Industrial network traffic logs for ICS anomaly detection.</div>
          <div class="dataset-use">Cyber + Physical Stream</div>
          <div><span class="badge-free">Free</span></div>
        </div>
        <div class="dataset-item">
          <div class="dataset-name">MIMII</div>
          <div class="dataset-desc">Published by Hitachi Research. Sound dataset for malfunctioning industrial machine
            investigation — pumps, fans, valves, slide rails recorded in real factory environments with background
            noise.</div>
          <div class="dataset-use">Audio Stream</div>
          <div><span class="badge-free">Free</span></div>
        </div>
        <div class="dataset-item">
          <div class="dataset-name">MVTec AD</div>
          <div class="dataset-desc">Industrial anomaly detection dataset with 15 categories of objects and textures.
            Used for visual anomaly detection fine-tuning on industrial components and surfaces.</div>
          <div class="dataset-use">Visual Stream</div>
          <div><span class="badge-free">Free</span></div>
        </div>"""
dataset_new = """        <div class="dataset-item">
          <div class="dataset-name">BATADAL</div>
          <div class="dataset-desc">Water treatment plant sensor telemetry with labeled system attack periods, valve manipulations, and pressure anomalies.</div>
          <div class="dataset-use">Physical Stream</div>
          <div><span class="badge-free">iTrust</span></div>
        </div>
        <div class="dataset-item">
          <div class="dataset-name">UNSW-NB15</div>
          <div class="dataset-desc">Comprehensive cyber dataset featuring hybrid real modern normal activities and synthetic contemporary attack behaviours.</div>
          <div class="dataset-use">Cyber Stream</div>
          <div><span class="badge-free">Cyber Range</span></div>
        </div>"""
text = text.replace(dataset_old, dataset_new)

# 9. Team
text = text.replace("Isolation Forest + Autoencoder cyber model", "Isolation Forest Bi-Modal Architecture")
text = text.replace("<li>Autoencoder audio pipeline</li>\\n", "")
text = text.replace("BATADAL + TON_IoT + MIMII data pipeline", "BATADAL + UNSW-NB15 data pipeline")

with codecs.open('c:\\Visionics OT\\otshield\\frontend\\index.html', 'w', 'utf-8') as f:
    f.write(text)
