<script>
export default {
  name: 'PredViz',
  props: {
    predicate: {
      type: Object,
    },
  },
};
</script>

<template>
  <span>{{predicate["attr"]}} {{predicate["op"]}} {{Number(predicate["val"]).toFixed(2)}}</span>
</template>

<style scoped>

</style>
