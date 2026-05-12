/**
 * its-hub plugin for OpenCode.ai
 *
 * Registers skills directory and injects bootstrap context.
 * Tells the agent where to find its-hub scripts.
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const ItsHubPlugin = async ({ client, directory }) => {
  const pluginRoot = path.resolve(__dirname, '../..');
  const skillsDir = path.join(pluginRoot, 'skills');
  const scriptsDir = path.join(pluginRoot, 'scripts');

  const getBootstrapContent = () => {
    return `<its-hub-plugin>
You have the its-hub inference-time scaling plugin installed.

**Available skills:**
- inference-scaling — improve LLM response quality via multiple candidates
- setup-guide — first-time configuration

**Script paths (use these instead of \${CLAUDE_PLUGIN_ROOT}):**
- Detection: ${scriptsDir}/its_detect.sh
- Scaling: ${scriptsDir}/its_scale.sh
- Server: ${scriptsDir}/its_server.sh

When skills reference \${CLAUDE_PLUGIN_ROOT}/scripts/..., substitute the paths above.
</its-hub-plugin>`;
  };

  return {
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      if (!config.skills.paths.includes(skillsDir)) {
        config.skills.paths.push(skillsDir);
      }
    },

    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
