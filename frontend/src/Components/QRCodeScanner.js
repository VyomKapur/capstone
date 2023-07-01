import React, { useState } from 'react';
import QrScanner from 'react-qr-scanner';

const QRCodeScanner = () => {
  const [qrCodeValue, setQRCodeValue] = useState('');

  const handleScan = (result) => {
    if (result) {
      setQRCodeValue(result);
    }
  };

  const handleError = (error) => {
    console.error(error);
  };

  return (
    <div>
      <QrScanner
        delay={300}
        onError={handleError}
        onScan={handleScan}
        style={{ width: '30%' }}
      />
      {qrCodeValue && <p>{qrCodeValue}</p>}
    </div>
  );
};

export default QRCodeScanner;
