/**
 * Basic usage example for sec-mem with OpenClaw
 */

const { SecMemPlugin } = require('@kevinzh1117/sec-mem');

async function main() {
  // Create plugin instance
  const plugin = new SecMemPlugin({
    collectionName: 'demo-memories',
    indexType: 'hnsw',
    embeddingDims: 512,
    storagePath: './demo-data',
    ollamaUrl: 'http://localhost:11434',
    ollamaEmbedModel: 'nomic-embed-text'
  });

  // Initialize
  console.log('Initializing sec-mem...');
  await plugin.initialize();
  console.log('✓ Ready\n');

  // Add some memories
  console.log('Adding memories...');
  
  const memories = [
    'I love working with Python and TypeScript',
    'My favorite database is PostgreSQL',
    'I prefer dark mode in all applications',
    'I work remotely from Tokyo, Japan'
  ];

  for (const memory of memories) {
    const result = await plugin.add(memory, 'user_001');
    console.log(`  Added: ${memory} (ID: ${result.id})`);
  }

  // Search memories
  console.log('\nSearching: "programming languages"');
  const search1 = await plugin.search('programming languages', 'user_001', 3);
  console.log('Results:', search1.results.map(r => r.memory));

  console.log('\nSearching: "where do I work"');
  const search2 = await plugin.search('where do I work', 'user_001', 3);
  console.log('Results:', search2.results.map(r => r.memory));

  // Get stats
  console.log('\nIndex stats:');
  const stats = await plugin.stats();
  console.log('  Type:', stats.indexType);
  console.log('  Vectors:', stats.numVectors);
  console.log('  Memory:', stats.memoryUsage.toFixed(2), 'MB');

  // Cleanup
  await plugin.shutdown();
  console.log('\n✓ Done');
}

main().catch(console.error);
