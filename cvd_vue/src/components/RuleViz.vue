<script>
import PredViz from "./PredViz.vue";

export default {
  name: "RuleViz",
  components: {
    PredViz,
  },
  props: {
    rule: {
      type: Object,
    }
  }
}
</script>

<template>
  <div class="mb-4">
    <div>IF</div>
    <div v-for="(p, i) in rule['premises']" :key="i" class="ml-2">
      <PredViz :predicate="p"/>
    </div>
    <div>THEN</div>
    <div class="ml-2">
      <PredViz :predicate="rule['consequence']"/>
    </div>
  </div>
</template>

<style scoped>

</style>
