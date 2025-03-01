// Initialize the Leaflet Map
const map = L.map("map").setView([48.27, -103.60], 9); // Centered on the middle of McKenzie and Williams County

// Convert the latitude and longtitude properly
function dmsToDecimal(dms) {
    const regex = /(\d+)[°\s]+(\d+)[']?\s*([\d.]+)["]?\s*([NSEW])/i;
    const parts = dms.match(regex);
    if (!parts) {
      return parseFloat(dms);
    }
    const degrees = parseInt(parts[1], 10);
    const minutes = parseInt(parts[2], 10);
    const seconds = parseFloat(parts[3]);
    const direction = parts[4].toUpperCase();
    let decimal = degrees + minutes / 60 + seconds / 3600;
    if (direction === "S" || direction === "W") {
      decimal *= -1;
    }
    return decimal;
  }

// Add OpenStreetMap Tile Layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);


// Fetch well data from the server
  fetch('/wells')
  .then(response => {
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    return response.json();
  })
  .then(data => {
    data.forEach(well => {
      let lat = (typeof well.latitude === 'string' && well.latitude.includes("°"))
                  ? dmsToDecimal(well.latitude)
                  : parseFloat(well.latitude);
      let lon = (typeof well.longitude === 'string' && well.longitude.includes("°"))
                  ? dmsToDecimal(well.longitude)
                  : parseFloat(well.longitude);
      
      if (!isNaN(lat) && !isNaN(lon)) {
        const marker = L.marker([lat, lon]).addTo(map);
        const popupContent = `
          <strong>${well.well_name}</strong><br>
          Operator: ${well.operator}<br>
          API: ${well.API}<br>
          County: ${well.county}<br>
          State: ${well.state}<br>
          Latitude: ${well.latitude}<br>
          Longitude: ${well.longitude}<br>
          Footages: ${well.footages}<br>
          Qtr_Qtr: ${well.Qtr_Qtr}<br>
          Section: ${well.section}<br>
          Township: ${well.township}<br>
          Range: ${well.range}<br>
          Well Status: ${well.well_status}<br>
          Well Type: ${well.well_type}<br>
          Closest City: ${well.closest_city}<br>
          Barrels Produced: ${well.barrels_produced}<br>
          MCF Gas Produced: ${well.mcf_gas_produced}
        `;
        marker.bindPopup(popupContent);
      }
    });
  })
  .catch(error => {
    console.error("Error fetching well data:", error);
  });