import { mockProducts } from "../data/mockData";

const STORAGE_KEY = "wisebite_recent_scans";
const MAX_SCANS = 10;

export function getRecentScans(): any[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      // Initialize with mock data if first time
      const initialScans = mockProducts.slice(0, 4);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(initialScans));
      return initialScans;
    }
    return JSON.parse(raw);
  } catch (err) {
    console.error("Failed to read recent scans from localStorage:", err);
    return mockProducts.slice(0, 4);
  }
}

export function saveRecentScan(product: any) {
  try {
    const scans = getRecentScans();
    
    // Check if the product already exists (by ID or exact name)
    const existingIndex = scans.findIndex(
      (p) => p.id === product.id || p.name === product.name
    );
    
    if (existingIndex > -1) {
      // Remove it so it can be moved to the front
      scans.splice(existingIndex, 1);
    }
    
    const newScans = [product, ...scans].slice(0, MAX_SCANS);
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newScans));
  } catch (err) {
    console.error("Failed to save recent scan to localStorage:", err);
  }
}
