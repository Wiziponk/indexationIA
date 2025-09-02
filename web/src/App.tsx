import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Wizard from "./pages/Wizard";
import Clustering from "./pages/Clustering";
import Library from "./pages/Library";
import Settings from "./pages/Settings";
import Datasets from "./pages/Datasets";

export default function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block px-4 py-2 hover:bg-muted ${isActive ? "bg-muted" : ""}`;

  return (
    <div className="flex min-h-screen">
      <nav className="w-48 border-r">
        <ul>
          <li>
            <NavLink to="/" end className={linkClass}>
              Home
            </NavLink>
          </li>
          <li>
            <NavLink to="/datasets" className={linkClass}>
              Datasets
            </NavLink>
          </li>
          <li>
            <NavLink to="/wizard" className={linkClass}>
              Dataset Wizard
            </NavLink>
          </li>
          <li>
            <NavLink to="/clustering" className={linkClass}>
              Clustering
            </NavLink>
          </li>
          <li>
            <NavLink to="/library" className={linkClass}>
              Library
            </NavLink>
          </li>
          <li>
            <NavLink to="/settings" className={linkClass}>
              Settings
            </NavLink>
          </li>
        </ul>
      </nav>
      <main className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/wizard" element={<Wizard />} />
          <Route path="/clustering" element={<Clustering />} />
          <Route path="/library" element={<Library />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}

