// Initialize the Leaflet Map
const map = L.map('map').setView([48.27, -103.60], 9); // Centered on the middle of McKenzie and Williams County

// Add OpenStreetMap Tile Layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: '&copy; OpenStreetMap contributors'}).addTo(map);

// Convert the latitude and longtitude properly
function dmsToDecimal(dms) {
    const parts = dms.match(/(\d+)° (\d+)' ([\d.]+)" (\w)/);
    let decimal = parseInt(parts[1]) + parseInt(parts[2]) / 60 + parseFloat(parts[3]) / 3600;
    if (parts[4] === "S" || parts[4] === "W") decimal *= -1;
    return decimal;
}

// Well Data
const wells = [
    { id: "W11745", operator: "RIM Operating Inc.", name: "Basic Game And Fish 34-3", api: "33-053-02102", county: "McKenzie County", state: "North Dakota", lat: "48° 5' 52.231\" N", lon: "103° 38' 42.770\" W", status: "Active", type: "Oil & Gas", barrels: 759, gas: 758 },
    { id: "W11920", operator: "RIM Operating Inc.", name: "Corps Of Engineers 31-10", api: "33-053-02148", county: "McKenzie County", state: "North Dakota", lat: "48° 5' 41.748\" N", lon: "103° 39' 23.940\" W", status: "Active", type: "Oil & Gas", barrels: 761, gas: 761 },
    { id: "W15358", operator: "Rim Operating Inc.", name: "Lewis And Clark 2-4H", api: "33-053-02556", county: "McKenzie County", state: "North Dakota", lat: "48° 6' 18.022\" N", lon: "103° 40' 12.900\" W", status: "Active", type: "Oil & Gas", barrels: 604, gas: 604 },
    { id: "W20197", operator: "Oasis Petroleum North America LLC", name: "Wade Federal 5300 21-30H", api: "33-053-03413", county: "McKenzie County", state: "North Dakota", lat: "48° 2' 49.420\" N", lon: "103° 36' 11.540\" W", status: "Inactive", type: "Oil & Gas", barrels: 0, gas: 0 },
    { id: "W20407", operator: "Oasis Petroleum North America LLC", name: "Chalmers 5300 31-19H", api: "33-053-03472", county: "McKenzie County", state: "North Dakota", lat: "48° 3' 26.470\" N", lon: "103° 36' 9.410\" W", status: "Abandoned", type: "Oil & Gas", barrels: 0, gas: 0 }
];

// Add Pins to the Map
wells.forEach(well => {
    const lat = dmsToDecimal(well.lat);
    const lon = dmsToDecimal(well.lon);

    const marker = L.marker([lat, lon]).addTo(map);
    marker.bindPopup(`
        <b>${well.name}</b><br>
        Operator: ${well.operator}<br>
        API: ${well.api}<br>
        Location: ${well.county}, ${well.state}<br>
        Latitude: ${well.lat}<br>
        Longitude: ${well.lon}<br>
        Well Status: ${well.status}<br>
        Type: ${well.type}<br>
        Barrels Produced: ${well.barrels}<br>
        MCF Gas Produced: ${well.gas}
    `);
});
