import './App.css';
import QRCodeScanner from "./Components/QRCodeScanner.js";

function App() {
  return (
    <div className="App">
      <nav className="navbar">
        <h1>QR Code Scanner</h1>
      </nav>
      <div className="content">
        <QRCodeScanner />
      </div>
    </div>
  );
}

export default App;
