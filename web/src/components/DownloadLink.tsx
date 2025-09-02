import React from "react";

interface Props {
  href: string;
}

export default function DownloadLink({ href }: Props) {
  const name = href.split("/").pop() ?? href;
  return (
    <a
      className="text-blue-600 underline"
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      {name}
    </a>
  );
}
