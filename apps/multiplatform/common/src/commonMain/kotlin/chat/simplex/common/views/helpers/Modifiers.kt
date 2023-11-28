package chat.simplex.common.views.helpers

import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.layout.offset
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.LayoutDirection
import kotlin.math.roundToInt

fun Modifier.badgeLayout() =
  layout { measurable, constraints ->
    val placeable = measurable.measure(constraints)

    // based on the expectation of only one line of text
    val minPadding = placeable.height / 4

    val width = maxOf(placeable.width + minPadding, placeable.height)
    layout(width, placeable.height) {
      placeable.place((width - placeable.width) / 2, 0)
    }
  }

@Composable
expect fun SwipeToDismissModifier(
  state: DismissState,
  directions: Set<DismissDirection> = setOf(DismissDirection.EndToStart, DismissDirection.StartToEnd),
  swipeDistance: Float,
): Modifier
