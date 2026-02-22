const path = require('path');

const defaultDistDir = '.next-local';
let configuredDistDir = process.env.NEXT_DIST_DIR || defaultDistDir;
if (path.isAbsolute(configuredDistDir)) {
  configuredDistDir = defaultDistDir;
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['three'],
  distDir: configuredDistDir,
};

module.exports = nextConfig;
