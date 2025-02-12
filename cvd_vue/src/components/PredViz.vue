<script>
export default {
  name: 'PredViz',
  props: {
    predicate: {
      type: Object,
    },
  },
  methods: {
    formatValue(val) {
      if (typeof val === 'number') {
        return Number(val).toFixed(2);
      }
      return val;
    },
  },
};
</script>

<template>
  <span>{{predicate["attr"]}} {{predicate["op"]}} {{formatValue(predicate["val"])}}</span>
</template>

<style scoped>

</style>
