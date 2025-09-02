import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { ToastProvider } from "./components/Toasts";
import { JobsProvider } from "./components/Jobs";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <JobsProvider>
          <App />
        </JobsProvider>
      </ToastProvider>
    </BrowserRouter>
  </React.StrictMode>
);

